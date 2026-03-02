import os
import numpy as np
import pandas as pd

diretorio = os.path.join(os.path.dirname(__file__), '../optuna_real_data/data'/camara'
os.makedirs(diretorio, exist_ok=True)

# Using complete range as in the author's script
anos = np.linspace(2001, 2023, num=2023-2001+1)

dt_votos_list = []
print("Downloading Votes (votacoesVotos)...")
for ano in anos:
    ano_int = int(ano)
    url = f'http://dadosabertos.camara.leg.br/arquivos/votacoesVotos/csv/votacoesVotos-{ano_int}.csv'
    try:
        dt_corrente = pd.read_csv(url, sep=';', usecols=['idVotacao',os.path.join(os.path.dirname(__file__), '../optuna_real_data/data'HoraVoto','voto','deputado_id','deputado_nome','deputado_siglaPartido','deputado_siglaUf'])
        dt_corrente = dt_corrente.loc[~pd.isnull(dt_corrente.voto)]
        dt_votos_list.append(dt_corrente)
        print(f"  [{ano_int}] Retrieved {len(dt_corrente)} votes.")
    except Exception as e:
        print(f"  [{ano_int}] Error: {e}")

dt_full = pd.concat(dt_votos_list)
dt_votos = dt_full.reset_index(drop=True)

dt_votacoes_list = []
print("\nDownloading Roll Calls (votacoes)...")
for ano in anos:
    ano_int = int(ano)
    url = f'http://dadosabertos.camara.leg.br/arquivos/votacoes/csv/votacoes-{ano_int}.csv'
    try:
        dt_corrente = pd.read_csv(url, sep=';', usecols=['id','siglaOrgao','aprovacao','votosSim','votosNao','votosOutros'])
        dt_votacoes_list.append(dt_corrente)
        print(f"  [{ano_int}] Retrieved {len(dt_corrente)} roll calls.")
    except Exception as e:
        print(f"  [{ano_int}] Error: {e}")

dt_full_votacoes = pd.concat(dt_votacoes_list)
dt_votacoes = dt_full_votacoes.reset_index(drop=True).rename(columns={"id": "idVotacao"})

print("\nMerging and Filtering...")
dt = dt_votos.merge(dt_votacoes, on='idVotacao', how='left')

# Original filters from the author
dt = dt\
    .loc[dt.siglaOrgao == 'PLEN']\
    .loc[dt.aprovacao.isin([0,1])]\
    .loc[dt.voto.isin(['Sim','Não'])]\
    .loc[~(dt.votosSim == 0)]\
    .loc[~(dt.votosNao == 0)]\
    .reset_index(drop=True)

print("Filtering applied. Unanimous votes and non-PLEN removed.")

# Automatic Name Standardization (Replaces the manual CSV output/input in the original)
print("Standardizing Deputado Names automatically...")
# Map each ID to its most frequent name
name_by_id = dt.groupby('deputado_id')['deputado_nome'].agg(lambda x: x.value_counts().index[0]).to_dict()
dt['deputado_nome'] = dt['deputado_id'].map(name_by_id)

# Map each Name to its most frequent ID
id_by_name = dt.groupby('deputado_nome')['deputado_id'].agg(lambda x: x.value_counts().index[0]).to_dict()
dt['deputado_id'] = dt['deputado_nome'].map(id_by_name)

# Extract Date
dt['Data'] = pd.to_datetime(dt.dataHoraVoto, format='%Y-%m-%dT%H:%M:%S').dt.date

# Save Final Dataset
caminho = os.path.join(diretorio, 'D03_Camara_Deputados_PLEN.csv')
print(f"Saving final dataset to: {caminho}")
dt[['Data','deputado_nome','deputado_siglaPartido','deputado_siglaUf','idVotacao','voto']].to_csv(caminho, index=False, encoding='utf-8')

print("Extraction complete!")
