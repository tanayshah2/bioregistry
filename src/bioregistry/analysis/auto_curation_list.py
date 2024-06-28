import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
import indra.literature.pubmed_client as pubmed_client

from bioregistry.bibliometrics import get_publications_df
from bioregistry.constants import EXPORT_ANALYSES

BASE_DIRECTORY = EXPORT_ANALYSES
AUTO_CURATION_DIRECTORY = BASE_DIRECTORY.joinpath("auto_curation")
AUTO_CURATION_DIRECTORY.mkdir(exist_ok=True, parents=True)

PMIDS = [
    38798726,
    38842319,
    38844844,
    38861393,
    38855461,
    38809036,
    38797350,
    38839789,
    38808228,
    38877053,
    38774187,
    38868520,
    38809035,
    38835641,
    38810119,
    38774441,
    38860781,
    38832799,
    38816381,
    38782708,

    38828463,
    38865724,
    38783072,
    38830240,
    38764687,
    38822436,
    38777815,
    38776476,
    38862393,
    38819076,
    38855044,
    38840122,
    38783951,
    38806173,
    38858638,
    38867492,
    38849215,
    38829948,
    38824044,
    38803621,

    38842642,
    38771531,
    38785476,
    38851652,
    38863761,
    38859848,
    38838406,
    38831198,
    38790220,
    38784852,
    38833830,
    38850201,
    38851806,
    38798682,
    38813836,
    38788903,
    38854090,
    38802199,
    38771013,
    38831986,
]


URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRPtP-tcXSx8zvhCuX6fqz_QvHowyAoDahnkixARk9rFTe0gfBN9GfdG6qTNQHHVL0i33XGSp_nV9XM/pub?output=csv")


def fetch_pubmed_papers():
    search_terms = ["database", "ontology", "resource", "vocabulary", "nomenclature"]
    all_pmids = []

    for term in search_terms:
        pmids = pubmed_client.get_ids(term, use_text_word=False, reldate=30)
        all_pmids.extend(pmids)

    all_pmids = list(set(all_pmids))
    if not all_pmids:
        click.echo(f"No PMIDs found for the last 30 days with the search terms: {search_terms}")
        return pd.DataFrame()

    papers = {}
    for chunk in [all_pmids[i:i + 200] for i in range(0, len(all_pmids), 200)]:
        papers.update(pubmed_client.get_metadata_for_ids(chunk))

    records = [
        {"pubmed": paper.get("pmid"), "title": paper.get("title"),
         "year": paper.get("publication_date", {}).get("year")}
        for paper in papers.values() if
        paper.get("title") and paper.get("pmid") and paper.get("publication_date", {}).get("year")
    ]
    return pd.DataFrame(records)

def fetch_metadata_for_pmids(pmids):
    papers = pubmed_client.get_metadata_for_ids(pmids)
    records = [
        {"pubmed": paper.get("pmid"), "title": paper.get("title")}
        for paper in papers.values() if paper.get("title") and paper.get("pmid")
    ]
    return pd.DataFrame(records)


def load_bioregistry_publications():
    click.echo("Loading bioregistry publications")
    df = get_publications_df()
    df = df[df.pubmed.notna() & df.title.notna()]
    df = df[["pubmed", "title", "year"]]
    df["label"] = True
    click.echo(f"Got {df.shape[0]} publications from the bioregistry")
    return df


def load_curation_data():
    click.echo("Downloading curation")
    df = pd.read_csv(URL)
    df["label"] = df["relevant"].map(_map_labels)
    df = df[["pubmed", "title", "label"]]
    click.echo(f"Got {df.label.notna().sum()} curated publications from Google Sheets")
    return df


def _map_labels(s: str):
    if s in {"1", "1.0", 1, 1.0}:
        return 1
    if s in {"0", "0.0", 0, 0.0}:
        return 0
    return None


def train_classifiers(x_train, y_train):
    classifiers = [
        ("rf", RandomForestClassifier()),
        ("lr", LogisticRegression()),
        ("dt", DecisionTreeClassifier()),
        ("svc", LinearSVC()),
        ("svm", SVC(kernel="rbf", probability=True))
    ]
    for _, clf in classifiers:
        clf.fit(x_train, y_train)
    return classifiers


def generate_meta_features(classifiers, x_train, y_train):
    meta_features = pd.DataFrame()
    for name, clf in classifiers:
        if hasattr(clf, "predict_proba"):
            predictions = cross_val_predict(clf, x_train, y_train, cv=5, method='predict_proba')[:, 1]
        else:
            predictions = cross_val_predict(clf, x_train, y_train, cv=5, method='decision_function')
        meta_features[name] = predictions
    return meta_features


def evaluate_meta_classifier(meta_clf, x_test_meta, y_test):
    y_pred = meta_clf.predict(x_test_meta)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, meta_clf.predict_proba(x_test_meta)[:, 1])
    return mcc, roc_auc


def predict_and_save(df, vectorizer, classifiers, meta_clf, filename):
    x_meta = pd.DataFrame()
    x_transformed = vectorizer.transform(df.title)
    for name, clf in classifiers:
        if hasattr(clf, "predict_proba"):
            x_meta[name] = clf.predict_proba(x_transformed)[:, 1]
        else:
            x_meta[name] = clf.decision_function(x_transformed)

    df['meta_score'] = meta_clf.predict_proba(x_meta)[:, 1]
    df = df.sort_values(by='meta_score', ascending=False)
    df.to_csv(AUTO_CURATION_DIRECTORY.joinpath(filename), sep="\t", index=False)
    click.echo(f"Writing predicted scores to {AUTO_CURATION_DIRECTORY.joinpath(filename)}")


@click.command()
def main() -> None:
    publication_df = load_bioregistry_publications()
    curation_df = load_curation_data()

    df = pd.concat([curation_df, publication_df])
    df["title"] = df["title"].str.slice(0, 20)

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(df.title)

    annotated_df = df[df.label.notna()]
    x = vectorizer.transform(annotated_df.title)
    y = annotated_df.label

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True)

    classifiers = train_classifiers(x_train, y_train)

    click.echo("scoring individual classifiers")
    scores = []
    for name, clf in classifiers:
        y_pred = clf.predict(x_test)
        try:
            mcc = matthews_corrcoef(y_test, y_pred)
        except ValueError as e:
            click.secho(f"{clf} failed to calculate MCC: {e}", fg="yellow")
            mcc = None
        try:
            if hasattr(clf, "predict_proba"):
                roc_auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, clf.decision_function(x_test))
        except AttributeError as e:
            click.secho(f"{clf} failed to calculate AUC-ROC: {e}", fg="yellow")
            roc_auc = None
        if not mcc and not roc_auc:
            continue
        scores.append((name, mcc or float("nan"), roc_auc or float("nan")))

    evaluation_df = pd.DataFrame(scores, columns=["classifier", "mcc", "auc_roc"]).round(3)
    evaluation_df.to_csv(AUTO_CURATION_DIRECTORY.joinpath("evaluation.tsv"), sep="\t", index=False)
    click.echo(tabulate(evaluation_df, showindex=False, headers=evaluation_df.columns))

    meta_features = generate_meta_features(classifiers, x_train, y_train)

    meta_clf = LogisticRegression()
    meta_clf.fit(meta_features, y_train)

    x_test_meta = pd.DataFrame()
    for name, clf in classifiers:
        if hasattr(clf, "predict_proba"):
            x_test_meta[name] = clf.predict_proba(x_test)[:, 1]
        else:
            x_test_meta[name] = clf.decision_function(x_test)

    mcc, roc_auc = evaluate_meta_classifier(meta_clf, x_test_meta, y_test)
    click.echo(f"Meta-Classifier MCC: {mcc}, AUC-ROC: {roc_auc}")

    random_forest_clf: RandomForestClassifier = classifiers[0][1]
    lr_clf: LogisticRegression = classifiers[1][1]
    importances_df = (
        pd.DataFrame(
            list(
                zip(
                    vectorizer.get_feature_names_out(),
                    vectorizer.idf_,
                    random_forest_clf.feature_importances_,
                    lr_clf.coef_[0],
                )
            ),
            columns=["word", "idf", "rf_importance", "lr_importance"],
        )
        .sort_values("rf_importance", ascending=False, key=abs)
        .round(4)
    )
    click.echo(tabulate(importances_df.head(15), showindex=False, headers=importances_df.columns))

    importance_path = AUTO_CURATION_DIRECTORY.joinpath("importances.tsv")
    click.echo(f"writing feature (word) importances to {importance_path}")
    importances_df.to_csv(importance_path, sep="\t", index=False)

    novel_df = df[~df.label.notna()][["pubmed", "title"]].copy()
    predict_and_save(novel_df, vectorizer, classifiers, meta_clf, "predictions_last_year.tsv")

    new_pub_df = fetch_pubmed_papers()
    if not new_pub_df.empty:
        predict_and_save(new_pub_df, vectorizer, classifiers, meta_clf, "predictions_last_month.tsv")

    # Fetch and score the provided list of PubMed IDs
    click.echo("Fetching and scoring the provided PubMed IDs")
    pmid_df = fetch_metadata_for_pmids(PMIDS)
    if not pmid_df.empty:
        predict_and_save(pmid_df, vectorizer, classifiers, meta_clf, "predictions_provided_pmids.tsv")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
