import argparse
import os
import pandas as pd
from tqdm import tqdm
from workflow import load_salt_remover, strip_salts, filter_organic, calc_descriptors


def main():
    parser = argparse.ArgumentParser(description="SMILES preprocessing pipeline")
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--salts', required=True, help='Salt definition file for RDKit SaltRemover')
    parser.add_argument('--output_dir', required=True, help='Directory to write outputs')
    parser.add_argument('--compute-descriptors', action='store_true', help='Whether to compute descriptors')
    parser.add_argument('--smiles-col', default='SMILES', help='Column name containing SMILES strings')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in input")
    remover = load_salt_remover(args.salts)
    tqdm.pandas()

    # Salt stripping
    df['smiles_stripped'] = df[args.smiles_col].progress_apply(lambda s: strip_salts(s, remover), desc='salt_strip')
    filtered1 = df.dropna(subset=['smiles_stripped'])
    filtered1.to_csv(os.path.join(args.output_dir, 'Preprocessed2_Saltstripped.csv'), index=False)

    # Organic filtering
    filtered2 = filtered1[filtered1['smiles_stripped'].progress_apply(filter_organic, desc='organic_filter')]
    filtered2.to_csv(os.path.join(args.output_dir, 'Preprocessed3_Organicselected.csv'), index=False)

    # Optional descriptor generation
    if args.compute_descriptors:
        desc_list = [calc_descriptors(smi) for smi in tqdm(filtered2['smiles_stripped'], desc='descriptors')]
        pd.concat([filtered2.reset_index(drop=True), pd.DataFrame(desc_list)], axis=1).to_csv(
            os.path.join(args.output_dir, 'Preprocessed4_DescriptorGen.csv'), index=False
        )


if __name__ == '__main__':
    main()
