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

    :param text: The `is_smiles` function is designed to check if a given input text is a valid SMILES
    (Simplified Molecular Input Line Entry System) string representing a chemical structure. The
    function attempts to create a molecule object from the input SMILES string using the RDKit library.
    If the molecule object
    :type text: str
    :return: The function `is_smiles` is checking if the input text is a valid SMILES (Simplified
    Molecular Input Line Entry System) string by attempting to create a molecule object using RDKit's
    `Chem.MolFromSmiles` function. If the function successfully creates a molecule object, it returns
    True, indicating that the input text is a valid SMILES string. If an exception occurs during the
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

    :param text: The function `is_multiple_smiles` takes a string as input and checks if the input
    contains a smiley face followed by a period. If the input string contains a smiley face and a
    period, the function returns `True`, otherwise it returns `False`
    :type text: str
    :return: The function `is_multiple_smiles` is checking if the input text contains a smiley face and
    a period. If the input text contains a smiley face and a period, it will return `True`. Otherwise,
    it will return `False`.
    """
    if is_smiles(text):
        return "." in text
    return False

def split_smiles( text: str) -> bool:
    """
    The function `split_smiles` takes a string as input and splits it by the period character.

    :param text: The `split_smiles` function you provided is currently returning a list of substrings
    obtained by splitting the input `text` at each occurrence of the period "." character
    :type text: str
    :return: The function `split_smiles` is returning a list of substrings obtained by splitting the
    input `text` string using the period "." as the delimiter.
    """
    return text.split(".")

def is_cas( text: str) -> Union[bool, None]:
    """
    The function `is_cas` checks if a given text matches a specific pattern for a CAS number.

    :param text: The `text` parameter is a string that is being checked for a specific pattern using a
    regular expression in the `is_cas` function. The function checks if the `text` matches the pattern
    `^\d{2,7}-\d{2}-\d$`, which represents a
    :type text: str
    :return: The function `is_cas` is returning a boolean value indicating whether the input `text`
    matches the specified pattern for a CAS number (Chemical Abstracts Service number). If the input
    `text` matches the pattern, the function returns `True`, indicating that it is a valid CAS number.
    If the input `text` does not match the pattern, the function returns `False`, indicating that
    """
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None

def largest_mol( smiles: str) -> str:
    """
    The function `largest_mol` takes a string of SMILES notation, splits it by '.', sorts the resulting
    list by length, removes any invalid SMILES strings, and returns the longest valid SMILES string.

    :param smiles: It looks like you have provided a code snippet for a function called `largest_mol`
    that takes a SMILES string as input. The function splits the input string by the '.' character,
    sorts the resulting list of substrings by length, and then removes any substrings that are not valid
    SM
    :type smiles: str
    :return: The function `largest_mol` takes a string `smiles` as input, which represents a list of
    SMILES strings separated by periods. It then splits the input string into individual SMILES strings,
    sorts them based on their length, and removes any invalid SMILES strings from the end of the list
    until a valid SMILES string is found. Finally, it returns the longest valid SMILES
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

    :param smiles: The parameter `smiles` in the `canonical_smiles` function is expected to be a string
    representing a SMILES (Simplified Molecular Input Line Entry System) notation of a chemical
    compound. The function attempts to convert this SMILES notation into a canonical form using the
    RDKit library in Python
    :type smiles: str
    :return: If the input SMILES string is valid and can be converted to a canonical SMILES
    representation, the function will return the canonical SMILES string. If the input SMILES string is
    invalid or cannot be processed, the function will return "Invalid SMILES string".
    """
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        return smi
    except Exception:
        return "Invalid SMILES string"

def tanimoto( s1: str, s2: str) -> str:
    """
    The function `tanimoto` calculates the Tanimoto similarity of two SMILES strings representing
    chemical compounds.

    :param s1: The `s1` parameter in the `tanimoto` function is a SMILES string representing a chemical
    compound. The function calculates the Tanimoto similarity between two SMILES strings by converting
    them into molecular objects and then generating Morgan fingerprints to compare their structural
    similarity
    :type s1: str
    :param s2: The `s2` parameter in the `tanimoto` function is a SMILES string representing a chemical
    compound. You can pass a valid SMILES string for a second chemical compound to calculate the
    Tanimoto similarity between the two compounds
    :type s2: str
    :return: The function `tanimoto` returns the Tanimoto similarity value between two SMILES strings.
    If there is an error in processing the SMILES strings, it will return the message "Error: Not a
    valid SMILES string".
    """
    """Calculate the Tanimoto similarity of two SMILES strings."""
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
    The function `query2smiles` takes a chemical compound name or SMILES string as input and retrieves
    the corresponding SMILES string using the PubChem API, handling cases where multiple SMILES strings
    are detected or no matching molecule is found.

    :param query: The `query` parameter in the `query2smiles` function is the input molecule name or
    identifier for which you want to retrieve the SMILES string. This can be a chemical compound name or
    identifier that you want to convert to its corresponding SMILES representation
    :type query: str
    :param url: The `url` parameter in the `query2smiles` function is a string that represents the URL
    template for querying the PubChem database to retrieve the SMILES string of a compound by its name.
    The `{}` placeholders in the URL string will be replaced with the query string and the specific
    endpoint, defaults to https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}
    :type url: str (optional)
    :return: The function `query2smiles` returns a SMILES string for a given chemical compound name or
    identifier. If the input query is already a valid SMILES string, it returns the same string. If the
    input query is not a SMILES string, it queries the PubChem database using the provided URL template
    to retrieve the SMILES string for the compound corresponding to the input query. If multiple SM
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

    :param query: The `query` parameter is the input molecule that you want to search for in PubChem to
    retrieve its CAS number. It can be either a chemical name or a SMILES string
    :type query: str
    :return: The function `query2cas` takes a query string, a URL for fetching CID information, and a
    URL for fetching data, and attempts to retrieve the CAS number for a molecule based on the query.
    """
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

    :param smiles: The `extract_function_groups` function takes a SMILES string representing a
    molecule as input and identifies functional groups present in the molecule based on a predefined
    dictionary of functional group patterns
    :type smiles: str
    :return: The `extract_function_groups` function returns a string indicating the functional
    groups present in a molecule represented by a SMILES string. The function checks for various
    functional groups defined in the `dict_fgs` dictionary within the input molecule using SMARTS
    patterns. If any functional groups are found in the molecule, the function constructs a sentence
    stating which functional groups are present. If multiple functional groups are found, the
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

    :param smiles_pair: The `smiles_pair` parameter in the `calculate_mole_similarity` function is a
    string that contains two SMILES strings separated by a period (`. `). These SMILES strings
    represent the chemical structures of two molecules that you want to compare for similarity
    :type smiles_pair: str
    :return: The function `calculate_mole_similarity` returns a message indicating the level of
    similarity between the molecules based on their Tanimoto similarity.
    The function `calculate_similarity` takes a pair of SMILES strings, calculates their Tanimoto similarity, and
    returns a message indicating the level of similarity between the molecules.
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

    :param smiles: The `calc_mole_weight` function takes a SMILES (Simplified Molecular Input Line Entry
    System) string as input and calculates the molecular weight of the corresponding molecule. If the
    input SMILES string is invalid and cannot be converted to a molecule object, the function will
    return "Invalid SMILES string
    :type smiles: str
    :return: The function `calc_mole_weight` returns the molecular weight calculated from the SMILES
    string input. If the input SMILES string is invalid and cannot be converted to a molecule object,
    the function returns the string "Invalid SMILES string".
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
    return mol_weight

def control_chemical_check( query: str) -> str:
    """
    The function `control_chemical_check` reads a CSV file, checks if a chemical query is in the data,
    and returns a message based on the result or similarity to known controlled chemicals.

    :param query: The `query` parameter in the `control_chemical_check` function is a string that
    represents either a SMILES notation or a CAS number of a chemical compound that you want to check
    against a list of controlled chemicals
    :type query: str
    :return: The function `control_chemical_check` returns a string message. If the query molecule or
    CAS number is found in the list of controlled chemicals, it returns a message stating that the
    molecule appears in the list of controlled chemicals. If the query is not found in the list, it
    attempts to get the SMILES representation of the CAS number and then calls the
    `similar_control_chem_check_run` function
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

    :param smiles: A SMILES string is a way to represent the structure of a chemical compound using
    ASCII characters. It stands for Simplified Molecular Input Line Entry System. It's a compact and
    human-readable way to represent chemical structures
    :type smiles: str
    :return: The function `run_similar_control_chem_check` returns a string message based on the
    similarity of the input SMILES string to known controlled chemicals in the provided data. The
    possible return messages are:
    1. If the input SMILES string is not valid: "Please input a valid SMILES string."
    2. If the maximum similarity is greater than 0.35: "{smiles} has
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

    :param cas_number: The `cas_number` parameter in the `run_explosive_check` function is used to
    represent the CAS number of a molecule. CAS number (Chemical Abstracts Service Registry Number) is a
    unique identifier assigned to chemical substances to provide a standardized way to identify and
    access chemical information. In this function, the CAS number is used to query PubChem for safety
    :return: The function `run_explosive_check` returns a message indicating whether the molecule with
    the given CAS number has an explosive GHS classification or not. If the molecule is classified as
    explosive, it returns "Molecule is explosive". If the molecule is not classified as explosive or if
    there is an error in the classification, it returns "Molecule is not known to be explosive" or an
    error message
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

    :param cas: The code you provided is a Python function that fetches safety data from PubChem for a
    given CAS number. It makes requests to the PubChem API to retrieve information about the specified
    chemical compound
    :return: The `get_safety_data` method returns a string representation of safety data fetched from
    PubChem for a given CAS number. The method fetches data from PubChem using the provided CAS number,
    processes the data by scraping specific sections, and then returns the extracted safety data in a
    structured format.
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

def search_papers_in_semantic_scholar(query:str,year:str=None,fields_of_study:str=None,return_num:int=10) -> str:
    '''Search for papers by keyword in semantic scholar

    :param str query: plain-text search query string.
    :param str year: (optional) restrict results to the given range of \
            publication year.
    :param list fields_of_study: (optional) restrict results to given \
            field-of-study list, using the s2FieldsOfStudy paper field.
    :returns: query results.
    '''
    results = sch.search_paper(query,year,fields_of_study,limit=return_num)
    ans = '\n'.join([item.title for item in results])
    return ans


def ddg_text_search(keywords:str,return_num:int=2) -> List[str]:
    """
    searching webpages related to text using google
    :param text: text for search
    :param return_num: numbers of webpages results to return
    :return: List of search results
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
                 control_chemical_check,run_similar_control_chem_check,run_explosive_check,get_safety_data,wikipedia_summary,search_papers_in_semantic_scholar,ddg_text_search]