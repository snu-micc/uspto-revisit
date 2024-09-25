import os
import logging
import json
import aiohttp
import asyncio
import nest_asyncio
import random
from tqdm import tqdm


from urllib.parse import quote
from chemspider_utils import get_smiles_from_ChemSpider_async
from file_io_utils import ensure_directory, save_smiles_dict
from json_name_fix_utils import is_valid_json, fix_json_string, fix_name

# Caching system for already fetched SMILES
nest_asyncio.apply()

global smiles_cache
smiles_cache = {}

if os.path.exists('smiles_cache.pkl'):
    with open('smiles_cache.pkl', 'rb') as f:
        smiles_cache = pickle.load(f)

async def exponential_backoff(attempt, max_delay=60):
    delay = min(max_delay, (2 ** attempt) + random.uniform(0, 1))
    await asyncio.sleep(delay)


async def fetch_smiles(session, url, semaphore, max_retries):
    async with semaphore: 
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429 or "ServerBusy" in await response.text():
                        logging.warning(f"[Busy] Server busy, retrying... (Attempt {attempt + 1}/{max_retries}) for URL: {url}")
                        await exponential_backoff(attempt)
                    else:
                        logging.error(f"[Error] Failed with status code: {response.status} for URL: {url}")
                        response.raise_for_status()
            except asyncio.TimeoutError:
                logging.error(f"[Timeout] Failed to fetch SMILES from {url} (Timeout)")
            except aiohttp.ClientError as e:
                logging.error(f"[ClientError] Failed to fetch SMILES from {url} [Error] {e}")
                if attempt < max_retries - 1:
                    await exponential_backoff(attempt)
            except Exception as e:
                logging.error(f"[Error] Failed to fetch SMILES from {url} [Error] {e}")
                if attempt < max_retries - 1:
                    await exponential_backoff(attempt)
    return None


async def get_smiles_from_pubchem(session, compound_name, semaphore, max_retries):
    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(compound_name)}/property/IsomericSMILES/JSON"
    try:
        response_text = await fetch_smiles(session, pubchem_url, semaphore, max_retries)
        if response_text:
            data = json.loads(response_text)
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and data['PropertyTable']['Properties']:
                return data['PropertyTable']['Properties'][0]['IsomericSMILES'], 'PubChem'
    except Exception as e:
        logging.error(f"[PubChem] Failed to fetch SMILES for {compound_name} from URL: {pubchem_url} [Error] {e}")
    return None, None


async def get_smiles_from_cir(session, compound_name, semaphore, max_retries):
    cir_url = f"http://cactus.nci.nih.gov/chemical/structure/{quote(compound_name)}/smiles"
    try:
        return await fetch_smiles(session, cir_url, semaphore, max_retries), 'CIR'
    except Exception as e:
        logging.error(f"[CIR] Failed to fetch SMILES for {compound_name} from URL: {cir_url} [Error] {e}")
    return None, None


async def get_smiles_from_opsin(session, compound_name, semaphore, max_retries):
    opsin_url = f"https://opsin.ch.cam.ac.uk/opsin/{quote(compound_name)}.smi"
    try:
        return await fetch_smiles(session, opsin_url, semaphore, max_retries), 'OPSIN'
    except Exception as e:
        logging.error(f"[OPSIN] Failed to fetch SMILES for {compound_name} from URL: {opsin_url} [Error] {e}")
    return None, None


async def get_smiles(session, compound_name, fix_name_bool, semaphore):
    compound_name = compound_name.replace("′","'")
    if not isinstance(compound_name, str):
        logging.warning(f"Warning: compound_name is not a string: {compound_name} (type: {type(compound_name)})")
        compound_name = str(compound_name)

    if fix_name_bool:
        if compound_name in ['ice', 'DCM', 'DMA','ether','brine','Pd/C','DMSO','LiAlH4']:
            return {'ice': 'O', 'DCM': 'C(Cl)Cl', 'DMA': 'CC(=O)N', 'ether': 'CCOCC', 'brine':'O.[Na+].[Cl-]','Pd/C':'Pd','DMSO':'CS(=O)C','LiAlH4':'[Li+].[AlH4-]'}.get(compound_name), 'Cache'
        else:
            compound_name = fix_name(compound_name)

    if compound_name in smiles_cache:
        return smiles_cache[compound_name], 'Cache'

    tasks = [
        get_smiles_from_opsin(session, compound_name, semaphore, max_retries=2),
    ]
    results = await asyncio.gather(*tasks)

    for result, source in results:
        if result:
            smiles_cache[compound_name] = result
            logging.info(f"Found SMILES for {compound_name} from {source}: {result}")
            return result, source
        
    logging.warning(f"Failed to find SMILES for {compound_name} in all sources.")
    return None, None


async def get_smiles_dict(response, session, fix_name_bool, semaphore):
    smiles_dict = {}
    problem_chemicals = []

    async def process_chemicals(chemicals_dict, category, fix_name_bool):
        tasks = {}
        for code, compound_name in chemicals_dict.items():
            tasks[code] = get_smiles(session, compound_name, fix_name_bool, semaphore)
        
        # print(tasks)
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for code, (smiles, source) in zip(tasks.keys(), results):
            if not smiles:
                fixed_name = fix_name(chemicals_dict[code])
                fixed_smiles, fixed_source = await get_smiles(session, fixed_name, True, semaphore)
                if not fixed_smiles:
                    problem_chemicals.append(f"{chemicals_dict[code]} ({category})")
                    smiles_dict[code] = f"[{chemicals_dict[code]} (NoSmi)]"
                else:
                    smiles_dict[code] = fixed_smiles
                    logging.info(f"Found SMILES for {chemicals_dict[code]} (Fixed Name) from {fixed_source}: {fixed_smiles}")
            else:
                smiles_dict[code] = smiles
                logging.info(f"Found SMILES for {chemicals_dict[code]} from {source}: {smiles}")

    if 'Reactants, Solvents, Catalysts' in response:
        await process_chemicals(response['Reactants, Solvents, Catalysts'], 'Reactant/Solvent/Catalyst', fix_name_bool)

    if 'Product' in response:
        await process_chemicals(response['Product'], 'Product', fix_name_bool)

    if problem_chemicals:
        logging.info(f"Problem chemicals: {problem_chemicals}")

    return smiles_dict


async def process_batch(json_responses, fix_name_bool, semaphore):
    smiles_dicts = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for json_response in json_responses:
            # Validate and fix JSON string
            if not is_valid_json(json_response):
                fixed_json = fix_json_string(json_response)
                if fixed_json and is_valid_json(fixed_json):
                    json_response = fixed_json
                else:
                    logging.error(f"Skipping invalid JSON response: {json_response}")
                    continue  # Skip this invalid JSON string

            # Process the valid JSON response
            response_dict = json.loads(json_response)
            tasks.append(get_smiles_dict(response_dict, session, fix_name_bool, semaphore))

        smiles_dicts = await asyncio.gather(*tasks)
    return smiles_dicts


async def process_no_smi_entry(code, compound_name, session, semaphore, smiles_dict,idx):
    tasks = [
        get_smiles_from_cir(session, compound_name, semaphore, max_retries=5),
        get_smiles_from_ChemSpider_async(compound_name),
        get_smiles_from_pubchem(session, compound_name, semaphore, max_retries=5)
    ]
    
    for task in asyncio.as_completed(tasks):
        try:
            smiles, source = await task
            if smiles:
                logging.info(f"[{idx}] Found SMILES for {compound_name} from {source}: {smiles}")
                smiles_dict[code] = smiles  # Update directly in the original dictionary
                return
        except Exception as e:
            logging.error(f"[{idx}] Error processing {compound_name}: {e}")

    logging.info(f"[{idx}] No SMILES found for {compound_name}")
    smiles_dict[code] = f"[{compound_name} (NoSmi)]"  # Mark as NoSmi if not found

async def process_batch_final(smiles_dict_list, session, semaphore):
    tasks = []
    for idx, smiles_dict in enumerate(smiles_dict_list):
        no_smi_entries = {key: value for key, value in smiles_dict.items() if "(NoSmi)" in value}
        for code, compound_name_with_no_smi in no_smi_entries.items():
            compound_name = compound_name_with_no_smi.replace("(NoSmi)", "").strip('[] ')
            tasks.append(process_no_smi_entry(code, compound_name, session, semaphore, smiles_dict, idx))
    
    await asyncio.gather(*tasks)

def calculate_no_smi_percentage(smiles_dict_list):
    total_entries = 0
    no_smi_entries = 0

    for smiles_dict in smiles_dict_list:
        total_entries += len(smiles_dict)
        no_smi_entries += sum(1 for value in smiles_dict.values() if "(NoSmi)" in value)
    
    no_smi_percentage = (no_smi_entries / total_entries) * 100 if total_entries > 0 else 0
    logging.info(f"Found {no_smi_entries} entries with NoSmi [{no_smi_percentage:.2f}%].")
    return no_smi_percentage

async def reprocess_no_smi(smiles_dict_file, output_file, session, semaphore, batch_size):
    with open(smiles_dict_file, 'r', encoding='utf-8-sig') as f:
        smiles_dict_list = json.load(f)

    total_batches = (len(smiles_dict_list) + batch_size - 1) // batch_size 

    with tqdm(total=total_batches, desc="Processing Batches", unit="batch") as pbar:
        for i in range(0, len(smiles_dict_list), batch_size):
            batch = smiles_dict_list[i:i + batch_size]
            await process_batch_final(batch, session, semaphore)
            no_smi_percentage = calculate_no_smi_percentage(smiles_dict_list)
            batch_number = i // batch_size + 1 
            print(f"[BATCH {batch_number}] {round(no_smi_percentage,2)}% of entries don't have a corresponding SMILES representation")
            logging.info(f"[BATCH {batch_number}] {round(no_smi_percentage, 2)}% of entries don't have a corresponding SMILES representation")

            # Save progress after each batch
            with open(output_file, 'w', encoding='utf-8-sig') as f:
                json.dump(smiles_dict_list, f, ensure_ascii=False, indent=2)

            logging.info(f"Batch {batch_number}/{total_batches} processed.")
            pbar.update(1)  # Update the progress bar after each batch
            
        logging.info("Processing completed")