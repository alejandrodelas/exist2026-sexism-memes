"""Model zoo: text (TF-IDF), physiological sensors, and combinations of both."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42


def word():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=60000)


def char():
    # character n-grams inside word boundaries: robust to OCR noise ("w0men", "mujerres") and spelling
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=120000)


def sensors(cols):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def features(kind, sensor_cols):
    parts = []
    if "word" in kind:
        parts.append(("word", word(), "text"))
    if "char" in kind:
        parts.append(("char", char(), "text"))
    if "sensor" in kind:
        parts.append(("sensor", sensors(sensor_cols), sensor_cols))
    weights = {"sensor": 0.5} if "sensor" in kind and len(parts) > 1 else None
    return ColumnTransformer(parts, transformer_weights=weights)


def zoo(sensor_cols):
    """name -> unfitted sklearn estimator taking the full DataFrame (text + sensor columns)."""
    lr = lambda bal=False: LogisticRegression(C=4, max_iter=3000, class_weight="balanced" if bal else None)
    svc = lambda bal=False: LinearSVC(C=0.5, class_weight="balanced" if bal else None)
    p = lambda kind, est: Pipeline([("features", features(kind, sensor_cols)), ("model", est)])
    m = {
        # ---- configurations from the original submission (May 2026)
        "word_tfidf_complement_nb": p("word", ComplementNB(alpha=0.5)),
        "char_tfidf_complement_nb": p("char", ComplementNB(alpha=0.5)),
        "word_tfidf_logreg": p("word", lr()),
        "char_tfidf_logreg": p("char", lr()),
        "word_char_tfidf_linearsvc": p("word+char", svc()),
        "word_char_tfidf_logreg": p("word+char", lr()),
        "word_char_sensor_linearsvc": p("word+char+sensor", svc()),
        "word_char_sensor_logreg": p("word+char+sensor", lr()),
        "sensor_extra_trees": p("sensor", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
                                                               n_jobs=-1, random_state=SEED)),
        # ---- improvements in this version
        "word_char_tfidf_logreg_balanced": p("word+char", lr(True)),
        "word_char_sensor_logreg_balanced": p("word+char+sensor", lr(True)),
    }
    # soft-voting ensemble: three different "views" of the meme vote with their probabilities
    m["ensemble_text_nb_lr_sensor"] = VotingClassifier(
        [("nb", p("word", ComplementNB(alpha=0.5))),
         ("lr", p("word+char", lr(True))),
         ("sens", p("sensor", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEED)))],
        voting="soft", weights=[1, 2, 1])
    return m


ORIGINAL_CHOICE = {"2.1": "word_tfidf_complement_nb", "2.2": "word_char_sensor_linearsvc", "2.3": "char_tfidf_logreg"}
