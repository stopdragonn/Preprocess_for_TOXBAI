import argparse
import os
import pandas as pd
import csv
from tqdm import tqdm
from rdkit import RDLogger
from workflow import (
    load_salt_smarts_list,
    strip_salts,
    filter_organic,
    calc_descriptors,
    parallel_series_apply,
)

# RDKit 경고 메시지 숨김
RDLogger.DisableLog('rdApp.*')

def load_input_file(input_path, smiles_col):
    ext = os.path.splitext(input_path)[-1].lower()
    try:
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        elif ext in ['.csv', '.txt']:
            with open(input_path, 'r', encoding='utf-8') as f:
                sample = f.read(2048)
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter

            df = pd.read_csv(
                input_path,
                encoding='utf-8',
                sep=delimiter,
                quoting=csv.QUOTE_MINIMAL,
                on_bad_lines='skip'
            )
        else:
            raise ValueError(f"지원되지 않는 입력 파일 형식입니다: {ext}")

        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in input file. 사용 가능한 컬럼: {df.columns.tolist()}")

        return df

    except Exception as e:
        raise RuntimeError(f"입력 파일 로딩 실패: {input_path}\n원인: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="SMILES preprocessing pipeline")
    parser.add_argument('--input', required=True, help='Input CSV/XLSX file')
    parser.add_argument('--salts', default='Salts.txt', help='Salt definition file (default: Salts.txt)')
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

    df = load_input_file(args.input, args.smiles_col)
    salt_mols = load_salt_smarts_list(args.salts)

    # Salt 제거
    print("\n[1] Stripping salts...")
    df['smiles_stripped'] = parallel_series_apply(
        df[args.smiles_col],
        lambda s: strip_salts(s, salt_mols),
        n_procs=1,
        desc='salt_strip',
    )
    filtered1 = df.dropna(subset=['smiles_stripped'])
    failed1 = df[df['smiles_stripped'].isna()]
    filtered1.to_csv(os.path.join(args.output_dir, 'Preprocessed2_Saltstripped.csv'), index=False)
    failed1.to_csv(os.path.join(args.output_dir, 'failed_saltstrip.csv'), index=False)

    print(f"총 {len(df)}개 중 salt 제거 성공: {len(filtered1)}개 / 실패: {len(failed1)}개")

    # Organic 필터링
    print("\n[2] Filtering organics...")
    mask_organic = parallel_series_apply(
        filtered1['smiles_stripped'],
        filter_organic,
        n_procs=args.n_procs,
        desc='organic_filter',
    )
    filtered2 = filtered1[mask_organic]
    filtered2.to_csv(os.path.join(args.output_dir, 'Preprocessed3_Organicselected.csv'), index=False)
    print(f"유기 필터링 통과: {len(filtered2)}개 / 제외: {len(filtered1) - len(filtered2)}개")

    # Descriptor 계산
    if args.compute_descriptors:
        print("\n[3] Calculating descriptors...")
        desc_df = parallel_series_apply(
            filtered2['smiles_stripped'],
            calc_descriptors,
            n_procs=args.n_procs,
            desc='descriptors',
        ).apply(pd.Series)

        output = pd.concat(
            [filtered2.reset_index(drop=True), desc_df.reset_index(drop=True)],
            axis=1,
        )
        output.to_csv(
            os.path.join(args.output_dir, 'Preprocessed4_DescriptorGen.csv'),
            index=False,
        )
        print(f"descriptor 계산 완료: {len(output)}개")

    print("\n✅ 전체 전처리 완료.")

if __name__ == '__main__':
    main()
