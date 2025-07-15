import argparse
import os
import pandas as pd
from tqdm import tqdm
from workflow import load_salt_remover, strip_salts, filter_organic, calc_descriptors


def main():
    parser = argparse.ArgumentParser(description="SMILES preprocessing pipeline")
    parser.add_argument('--input', required=True, help='Input CSV with a SMILES column named "SMILES"')
    parser.add_argument('--salts', required=True, help='Salt definition file for RDKit SaltRemover')
    parser.add_argument('--output_dir', required=True, help='Directory to write outputs')
    parser.add_argument('--compute-descriptors', action='store_true', help='Whether to compute descriptors')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    remover = load_salt_remover(args.salts)

    # Salt stripping
    df['smiles_stripped'] = df['SMILES'].apply(lambda s: strip_salts(s, remover))
    filtered1 = df.dropna(subset=['smiles_stripped'])
    filtered1.to_csv(os.path.join(args.output_dir, 'Preprocessed2_Saltstripped.csv'), index=False)

    # Organic filtering
    filtered2 = filtered1[filtered1['smiles_stripped'].apply(filter_organic)]
    filtered2.to_csv(os.path.join(args.output_dir, 'Preprocessed3_Organicselected.csv'), index=False)

    # Optional descriptor generation
    if args.compute_descriptors:
        desc_list = [calc_descriptors(smi) for smi in tqdm(filtered2['smiles_stripped'])]
        pd.concat([filtered2.reset_index(drop=True), pd.DataFrame(desc_list)], axis=1).to_csv(
            os.path.join(args.output_dir, 'Preprocessed4_DescriptorGen.csv'), index=False
        )


if __name__ == '__main__':
    main()
