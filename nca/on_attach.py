import textwrap


def on_attach():
    # Metadata placeholders
    meta = {"Date": "2024-01-01", "Title": "Necessary Condition Analysis", "Version": "4.0.4"}
    year = meta["Date"].split("-", maxsplit=1)[0]
    title = meta["Title"]

    msg1 = f"Dul, J. {year}."
    msg2 = f"{title}."
    msg3 = f"Python Package Version {meta['Version']}.\n"
    msg4 = "URL: https://cran.r-project.org/web/packages/NCA/"

    msg5 = "This package is based on:"
    msg6 = 'Dul, J. (2016) "Necessary Condition Analysis (NCA):'
    msg7 = "Logic and Methodology of 'Necessary but Not Sufficient' Causality.\""
    msg8 = "Organizational Research Methods 19(1), 10-52."
    msg9 = "https://journals.sagepub.com/doi/full/10.1177/1094428115584005"

    msg10 = 'Dul, J. (2020) "Conducting Necessary Condition Analysis"'
    msg11 = "SAGE Publications, ISBN: 9781526460141"
    msg12 = "https://uk.sagepub.com/en-gb/eur/conducting-necessary-condition-"
    msg13 = "analysis-for-business-and-management-students/book262898"

    msg14 = "Dul, J., van der Laan, E., & Kuik, R. (2020)."
    msg15 = 'A statistical significance test for Necessary Condition Analysis."'
    msg16 = "Organizational Research Methods, 23(2), 385-395."
    msg17 = "https://journals.sagepub.com/doi/10.1177/1094428118795272"

    msg18 = "A BibTeX entry is provided by:"
    msg19 = "citation('NCA')"

    msg20 = "A quick start guide can be found here:"
    msg21 = "https://repub.eur.nl/pub/78323/"
    msg22 = "or"
    msg23 = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2624981"

    msg24 = "For general information about NCA see :"
    msg25 = "https://www.erim.nl/nca"

    def wrap(text, indent=0, exdent=0):
        return textwrap.fill(
            text, width=80, initial_indent=" " * indent, subsequent_indent=" " * exdent
        )

    print("\nPlease cite the NCA package as:\n")
    print(wrap(msg1, 2, 2))
    print(wrap(msg2, 2, 2))
    print(wrap(msg3, 2, 2))
    print(wrap(msg4, 2, 2))
    print("\n")
    print(wrap(msg5, 0, 2))
    print(wrap(msg6, 2, 2))
    print(wrap(msg7, 2, 2))
    print(wrap(msg8, 2, 2))
    print(wrap(msg9, 2, 2))
    print("and")
    print(wrap(msg10, 2, 2))
    print(wrap(msg11, 2, 2))
    print(wrap(msg12, 2, 2))
    print(wrap(msg13, 2, 2))
    print("and")
    print(wrap(msg14, 2, 2))
    print(wrap(msg15, 2, 2))
    print(wrap(msg16, 2, 2))
    print(wrap(msg17, 2, 2))
    print("\n")
    print(wrap(msg18, 0, 2))
    print(wrap(msg19, 2, 2))
    print("\n")
    print(wrap(msg20, 0, 2))
    print(wrap(msg21, 2, 2))
    print(wrap(msg22, 2, 2))
    print(wrap(msg23, 2, 2))
    print("\n")
    print(wrap(msg24, 0, 2))
    print(wrap(msg25, 2, 2))
    print("\n")
