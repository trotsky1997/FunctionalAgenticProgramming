import json
import re
from typing import List, Union
import os
import wikipedia
from semanticscholar import SemanticScholar
sch = SemanticScholar()
from langchain_community.utilities import SearxSearchWrapper

searcher = SearxSearchWrapper(
    searx_host="https://searx.makelovenowar.win/search")

try:
    __import__("rdkit")
except ImportError:
    os.system("python -m pip install rdkit")

try:
    __import__("langchain")
except ImportError:
    os.system("python -m pip install langchain")
    os.system("python -m pip install langchain-community")


import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from langchain_community.utilities import SearxSearchWrapper

safety_summary_prompt = (
    "Your task is to parse through the data provided and provide a summary of important health, laboratory, and environemntal safety information."
    'Focus on answering the following points, and follow the format "Name: description".'
    "Operator safety: Does this substance represent any danger to the person handling it? What are the risks? What precautions should be taken when handling this substance?"
    "GHS information: What are the GHS signal (hazard level: dangerous, warning, etc.) and GHS classification? What do these GHS classifications mean when dealing with this substance?"
    "Environmental risks: What are the environmental impacts of handling this substance."
    "Societal impact: What are the societal concerns of this substance? For instance, is it a known chemical weapon, is it illegal, or is it a controlled substance for any reason?"
    "For each point, use maximum two sentences. Use only the information provided in the paragraph below."
    "If there is not enough information in a category, you may fill in with your knowledge, but explicitly state so."
    "Here is the information:{data}"
)

summary_each_data = (
    "Please summarize the following, highlighting important information for health, laboratory and environemntal safety."
    "Do not exceed {approx_length} characters. The data is: {data}"
)


from langchain.tools import BaseTool
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

"""Wrapper for RXN4Chem functionalities."""

import re
from time import sleep
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.tools import BaseTool

# from rxn4chemistry import RXN4ChemistryWrapper  # type: ignore

# clintox = pd.read_csv(
#                     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
#                 )
cw_df = pd.read_csv(
    "https://raw.githubusercontent.com/trotsky1997/Notes/master/chem_wep_smi.csv"
)
url_cid = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/cids/JSON"
url_data = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"


# class Tools:

def is_smiles( text: str) -> bool:
    """
    The function `is_smiles` checks if a given text is a valid SMILES notation using the RDKit library
    in Python.
    """
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False

def is_multiple_smiles( text: str) -> bool:
    """
    The function `is_multiple_smiles` checks if the input text contains a smiley face and a period.
    """
    if is_smiles(text):
        return "." in text
    return False

def split_smiles( text: str) -> bool:
    """
    The function `split_smiles` takes a string as input and splits it by the period character.
    """
    return text.split(".")

def is_cas( text: str) -> Union[bool, None]:
    """
    The function `is_cas` checks if a given text matches a specific pattern for a CAS number.
    """
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None

def largest_mol( smiles: str) -> str:
    """
    The function `largest_mol` takes a string of SMILES notation, splits it by '.', sorts the resulting
    list by length, removes any invalid SMILES strings, and returns the longest valid SMILES string.
    """
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    while not is_smiles(ss[-1]):
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]

def canonical_smiles( smiles: str) -> str:
    """
    The function `canonical_smiles` takes a SMILES string as input and returns its canonical form if
    valid, otherwise it returns "Invalid SMILES string".
    """
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        return smi
    except Exception:
        return "Invalid SMILES string"

def tanimoto( s1: str, s2: str) -> str:
    """
    Calculate the Tanimoto similarity of two SMILES strings.
    """
    try:
        mol1 = Chem.MolFromSmiles(s1)
        mol2 = Chem.MolFromSmiles(s2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except (TypeError, ValueError, AttributeError):
        return "Error: Not a valid SMILES string"

def query2smiles(
    query: str,
) -> str:
    """
    Input a molecule name, returns SMILES.
    """
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    if is_smiles(query):
        if not is_multiple_smiles(query):
            return query
        else:
            return "Multiple SMILES strings detected, input one molecule at a time."

    if url is None:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    r = requests.get(url.format(query, "property/IsomericSMILES/JSON"))
    # convert the response to a json object
    data = r.json()
    # return the SMILES string
    try:
        smi = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except KeyError:
        return "Could not find a molecule matching the text. One possible cause is that the input is incorrect, input one molecule at a time."
    return str(Chem.CanonSmiles(largest_mol(smi)))

def query2cas( query: str) -> str:
    """
    The function `query2cas` takes a molecule query, retrieves its PubChem CID, and returns its CAS
    number if available.
    """
    url_cid = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/cids/JSON")
    url_data = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
        )
    try:
        mode = "name"
        if is_smiles(query):
            if is_multiple_smiles(query):
                raise ValueError(
                    "Multiple SMILES strings detected, input one molecule at a time."
                )
            mode = "smiles"
        url_cid = url_cid.format(mode, query)
        cid = requests.get(url_cid).json()["IdentifierList"]["CID"][0]
        url_data = url_data.format(cid)
        data = requests.get(url_data).json()
    except (requests.exceptions.RequestException, KeyError):
        raise ValueError("Invalid molecule input, no Pubchem entry")

    try:
        for section in data["Record"]["Section"]:
            if section.get("TOCHeading") == "Names and Identifiers":
                for subsection in section["Section"]:
                    if subsection.get("TOCHeading") == "Other Identifiers":
                        for subsubsection in subsection["Section"]:
                            if subsubsection.get("TOCHeading") == "CAS":
                                return subsubsection["Information"][0]["Value"][
                                    "StringWithMarkup"
                                ][0]["String"]
    except KeyError:
        raise ValueError("Invalid molecule input, no Pubchem entry")

    raise ValueError("CAS number not found")

def extract_function_groups( smiles: str) -> str:
    """
    The function `extract_function_groups` takes a molecular SMILES string as input and identifies
    functional groups present in the molecule based on predefined SMARTS patterns.
    """
    dict_fgs = {
        "furan": "o1cccc1",
        "aldehydes": " [CX3H1](=O)[#6]",
        "esters": " [#6][CX3](=O)[OX2H0][#6]",
        "ketones": " [#6][CX3](=O)[#6]",
        "amides": " C(=O)-N",
        "thiol groups": " [SH]",
        "alcohol groups": " [OH]",
        "methylamide": "*-[N;D2]-[C;D3](=O)-[C;D1;H3]",
        "carboxylic acids": "*-C(=O)[O;D1]",
        "carbonyl methylester": "*-C(=O)[O;D2]-[C;D1;H3]",
        "terminal aldehyde": "*-C(=O)-[C;D1]",
        "amide": "*-C(=O)-[N;D1]",
        "carbonyl methyl": "*-C(=O)-[C;D1;H3]",
        "isocyanate": "*-[N;D2]=[C;D2]=[O;D1]",
        "isothiocyanate": "*-[N;D2]=[C;D2]=[S;D1]",
        "nitro": "*-[N;D3](=[O;D1])[O;D1]",
        "nitroso": "*-[N;R0]=[O;D1]",
        "oximes": "*=[N;R0]-[O;D1]",
        "Imines": "*-[N;R0]=[C;D1;H2]",
        "terminal azo": "*-[N;D2]=[N;D2]-[C;D1;H3]",
        "hydrazines": "*-[N;D2]=[N;D1]",
        "diazo": "*-[N;D2]#[N;D1]",
        "cyano": "*-[C;D2]#[N;D1]",
        "primary sulfonamide": "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
        "methyl sulfonamide": "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
        "sulfonic acid": "*-[S;D4](=O)(=O)-[O;D1]",
        "methyl ester sulfonyl": "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
        "methyl sulfonyl": "*-[S;D4](=O)(=O)-[C;D1;H3]",
        "sulfonyl chloride": "*-[S;D4](=O)(=O)-[Cl]",
        "methyl sulfinyl": "*-[S;D3](=O)-[C;D1]",
        "methyl thio": "*-[S;D2]-[C;D1;H3]",
        "thiols": "*-[S;D1]",
        "thio carbonyls": "*=[S;D1]",
        "halogens": "*-[#9,#17,#35,#53]",
        "t-butyl": "*-[C;D4]([C;D1])([C;D1])-[C;D1]",
        "tri fluoromethyl": "*-[C;D4](F)(F)F",
        "acetylenes": "*-[C;D2]#[C;D1;H]",
        "cyclopropyl": "*-[C;D3]1-[C;D2]-[C;D2]1",
        "ethoxy": "*-[O;D2]-[C;D2]-[C;D1;H3]",
        "methoxy": "*-[O;D2]-[C;D1;H3]",
        "side-chain hydroxyls": "*-[O;D1]",
        "ketones": "*=[O;D1]",
        "primary amines": "*-[N;D1]",
        "nitriles": "*#[N;D1]",
    }

    def is_fg_in_mol(mol, fg):
        fgmol = Chem.MolFromSmarts(fg)
        mol = Chem.MolFromSmiles(mol.strip())
        return len(Chem.Mol.GetSubstructMatches(mol, fgmol, uniquify=True)) > 0

    try:
        fgs_in_molec = [
            name for name, fg in dict_fgs.items() if is_fg_in_mol(smiles, fg)
        ]
        if len(fgs_in_molec) > 1:
            return f"This molecule contains {', '.join(fgs_in_molec[:-1])}, and {fgs_in_molec[-1]}."
        else:
            return f"This molecule contains {fgs_in_molec[0]}."
    except:
        return "Wrong argument. Please input a valid molecular SMILES."

def calculate_mole_similarity( smiles_pair: str) -> str:
    """
    The `calculate_mole_similarity` function takes a pair of SMILES strings, calculates their
    Tanimoto similarity, and provides a message indicating the level of similarity between the
    molecules.
    """
    smi_list = smiles_pair.split(".")
    if len(smi_list) != 2:
        return "Input error, please input two smiles strings separated by '.'"
    else:
        smiles1, smiles2 = smi_list

    similarity = tanimoto(smiles1, smiles2)

    if isinstance(similarity, str):
        return similarity

    sim_score = {
        0.9: "very similar",
        0.8: "similar",
        0.7: "somewhat similar",
        0.6: "not very similar",
        0: "not similar",
    }
    if similarity == 1:
        return "Error: Input Molecules Are Identical"
    else:
        val = sim_score[
            max(key for key in sim_score.keys() if key <= round(similarity, 1))
        ]
        message = f"The Tanimoto similarity between {smiles1} and {smiles2} is {round(similarity, 4)},\
        indicating that the two molecules are {val}."
    return message

def calc_mole_weight( smiles: str) -> str:
    """
    The function calculates the molecular weight of a compound based on its SMILES string
    representation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
    return mol_weight

def control_chemical_check( query: str) -> str:
    """
    The function `control_chemical_check` reads a CSV file, checks if a molecule or CAS number  is in the data,    
    and returns a message based on the result or similarity to known controlled chemicals.
    """
    try:
        if is_smiles(query):
            query_esc = re.escape(query)
            found = (
                cw_df["smiles"]
                .astype(str)
                .str.contains(f"^{query_esc}$", regex=True)
                .any()
            )
        else:
            found = (
                cw_df["cas"]
                .astype(str)
                .str.contains(f"^\({query}\)$", regex=True)
                .any()
            )
        if found:
            return (
                f"The molecule {query} appears in a list of "
                "controlled chemicals."
            )
        else:
            # Get smiles of CAS number
            try:
                smi = query2smiles(query)
            except ValueError as e:
                return str(e)
            # Check similarity to known controlled chemicals
            return run_similar_control_chem_check(smi)

    except Exception as e:
        return f"Error: {e}"

def run_similar_control_chem_check( smiles: str) -> str:
    """
    The function `run_similar_control_chem_check` reads a CSV file, calculates the maximum similarity
    between a given SMILES string and known chemicals, and returns a message based on the similarity
    threshold.
    """

    try:
        if not is_smiles(smiles):
            return "Please input a valid SMILES string."

        max_sim = cw_df["smiles"].apply(lambda x: tanimoto(smiles, x)).max()
        if max_sim > 0.35:
            return (
                f"{smiles} has a high similarity "
                f"({max_sim:.4}) to a known controlled chemical."
            )
        else:
            return (
                f"{smiles} has a low similarity "
                f"({max_sim:.4}) to a known controlled chemical."
                "This is substance is safe, you may proceed with the task."
            )
    except:
        return "Tool error."

def run_explosive_check(cas_number:str)  -> str:
    """
    The `run_explosive_check` function checks if a molecule has an explosive GHS classification using
    PubChem based on the provided CAS number.
    """
    pubchem_data = {}

    def _fetch_pubchem_data(cas_number):
        """Fetch data from PubChem for a given CAS number, or use cached data if it's already been fetched."""
        if cas_number not in pubchem_data:
            try:
                url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/cids/JSON"
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{requests.get(url1).json()['IdentifierList']['CID'][0]}/JSON"
                r = requests.get(url)
                pubchem_data[cas_number] = r.json()
            except:
                return "Invalid molecule input, no Pubchem entry."
        return pubchem_data[cas_number]

    def ghs_classification(text):
        """Gives the ghs classification from Pubchem. Give this tool the name or CAS number of one molecule."""
        if is_smiles(text):
            return "Please input a valid CAS number."
        data = _fetch_pubchem_data(text)
        if isinstance(data, str):
            return "Molecule not found in Pubchem."
        try:
            for section in data["Record"]["Section"]:
                if section.get("TOCHeading") == "Chemical Safety":
                    ghs = [
                        markup["Extra"]
                        for markup in section["Information"][0]["Value"][
                            "StringWithMarkup"
                        ][0]["Markup"]
                    ]
                    if ghs:
                        return ghs
        except (StopIteration, KeyError):
            return None

    # first check if the input is a CAS number
    if is_smiles(cas_number):
        return "Please input a valid CAS number."
    cls = ghs_classification(cas_number)
    if cls is None:
        return (
            "Explosive Check Error. The molecule may not be assigned a GHS rating. "
        )
    if "Explos" in str(cls) or "explos" in str(cls):
        return "Molecule is explosive"
    else:
        return "Molecule is not known to be explosive"

def get_safety_data(cas:str)  -> str:
    """
    The function `get_safety_data` fetches safety data from PubChem for a given CAS number and organizes
    it into specific categories.
    """
    pubchem_data = {}

    def _fetch_pubchem_data(cas_number):
        """Fetch data from PubChem for a given CAS number, or use cached data if it's already been fetched."""
        if cas_number not in pubchem_data:
            try:
                url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/cids/JSON"
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{requests.get(url1).json()['IdentifierList']['CID'][0]}/JSON"
                r = requests.get(url)
                pubchem_data[cas_number] = r.json()
            except:
                return "Invalid molecule input, no Pubchem entry."
        return pubchem_data[cas_number]

    @staticmethod
    def _scrape_pubchem(data, heading1, heading2, heading3):
        try:
            filtered_sections = []
            for section in data["Record"]["Section"]:
                toc_heading = section.get("TOCHeading")
                if toc_heading == heading1:
                    for section2 in section["Section"]:
                        if section2.get("TOCHeading") == heading2:
                            for section3 in section2["Section"]:
                                if section3.get("TOCHeading") == heading3:
                                    filtered_sections.append(section3)
            return filtered_sections
        except:
            return None

    data = _fetch_pubchem_data(cas)
    safety_data = []

    iterations = [
        (
            [
                "Health Hazards",
                "GHS Classification",
                "Hazards Summary",
                "NFPA Hazard Classification",
            ],
            "Safety and Hazards",
            "Hazards Identification",
        ),
        (
            ["Explosive Limits and Potential", "Preventive Measures"],
            "Safety and Hazards",
            "Safety and Hazard Properties",
        ),
        (
            [
                "Inhalation Risk",
                "Effects of Long Term Exposure",
                "Personal Protective Equipment (PPE)",
            ],
            "Safety and Hazards",
            "Exposure Control and Personal Protection",
        ),
        (
            ["Toxicity Summary", "Carcinogen Classification"],
            "Toxicity",
            "Toxicological Information",
        ),
    ]

    for items, header1, header2 in iterations:
        safety_data.extend(
            [_scrape_pubchem(data, header1, header2, item)] for item in items
        )

    return str(safety_data)


def wikipedia_summary(topic:str,return_num:int=2) -> List[str]:
    """
    The function `wikipedia_summary` takes a topic as input and returns a list of summaries from
    Wikipedia related to that topic. The optional parameter `return_num` specifies the number of
    summaries to return, with a default value of 3.
    """

    k = wikipedia.search(topic)
    ans = []
    for i in k:
        if len(ans) >= return_num:
            break
        try :
            summ = wikipedia.summary(i)
            ans.append(summ)
        except:
            pass
    return ans

def search_papers_in_semantic_scholar(query:str,return_num:int=10) -> str:
    '''Search for papers by keyword in semantic scholar

    :param str query: plain-text search query string.
    :returns: query results.
    '''
    results = sch.search_paper(query,limit=return_num)
    ans = '\n'.join([item.title for item in results])
    return ans


def ddg_text_search(keywords:str,return_num:int=2) -> List[str]:
    """
    searching webpages related to text using google, keywords better in English
    """
    results = searcher.results(
        keywords,
        num_results=return_num,
        time_range="year",
        enabled_engines=["bing", "google", "duckduckgo", "qwant", "yahoo", '	wikipedia',
                         'wikidata', 'wolframalpha', 'semantic scholar', 'google scholar', 'arxiv', 'pubmed']
    )

    return json.dumps(results)

# def ddg_keyword_ask(text:str,return_num:str=2) -> List[str]:
#     """
#     get useful keyword explanations from google
#     :param text: single keyword for search
#     :param return_num: numbers of descriptions to return
#     :return: List of search results
#     """
#     reuslts = list(ddgs.answers(text))
#     return reuslts[:return_num]


all_functions = [is_smiles,is_multiple_smiles,split_smiles,is_cas,largest_mol,canonical_smiles,
                 tanimoto,query2smiles,query2cas,extract_function_groups,calculate_mole_similarity,calc_mole_weight,
                 control_chemical_check,run_similar_control_chem_check,run_explosive_check,wikipedia_summary,search_papers_in_semantic_scholar,ddg_text_search]
