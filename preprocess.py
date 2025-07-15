import argparse
import os
import pandas as pd
from tqdm import tqdm
from workflow import (
    load_salt_remover,
    strip_salts,
    filter_organic,
    calc_descriptors,
    parallel_series_apply,
)


def main():
    """Command-line interface for running the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="SMILES preprocessing pipeline")
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument(
        '--salts',
        default='Salts.txt',
        help='Salt definition file for RDKit SaltRemover (default: Salts.txt)'
    )
    parser.add_argument('--output_dir', required=True, help='Directory to write outputs')
    parser.add_argument('--compute-descriptors', action='store_true', help='Whether to compute descriptors')
    parser.add_argument('--smiles-col', default='SMILES', help='Column name containing SMILES strings')
    parser.add_argument('--n-procs', type=int, default=1, help='Number of parallel processes')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.isfile(args.salts):
        raise FileNotFoundError(f"Salt definition file not found: {args.salts}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in input")
    remover = load_salt_remover(args.salts)
    # Salt stripping
    df['smiles_stripped'] = parallel_series_apply(
        df[args.smiles_col],
        lambda s: strip_salts(s, remover),
        n_procs=args.n_procs,
        desc='salt_strip',
    )
    filtered1 = df.dropna(subset=['smiles_stripped'])
    filtered1.to_csv(os.path.join(args.output_dir, 'Preprocessed2_Saltstripped.csv'), index=False)

    # Organic filtering
    mask_organic = parallel_series_apply(
        filtered1['smiles_stripped'],
        filter_organic,
        n_procs=args.n_procs,
        desc='organic_filter',
    )
    filtered2 = filtered1[mask_organic]
    filtered2.to_csv(os.path.join(args.output_dir, 'Preprocessed3_Organicselected.csv'), index=False)

    # Optional descriptor generation
    if args.compute_descriptors:
        desc_df = parallel_series_apply(
            filtered2['smiles_stripped'],
            calc_descriptors,
            n_procs=args.n_procs,
            desc='descriptors',
        ).apply(pd.Series)
        pd.concat(
            [filtered2.reset_index(drop=True), desc_df.reset_index(drop=True)],
            axis=1,
        ).to_csv(
            os.path.join(args.output_dir, 'Preprocessed4_DescriptorGen.csv'),
            index=False,
        )


if __name__ == '__main__':
    main()
