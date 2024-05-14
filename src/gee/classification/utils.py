import ee


def sklearn_to_gee_kwds(d_kwds_sklearn, verbose=1):
    d_kwds_gee = {}
    for k, v in d_kwds_sklearn.items():
        if k == "n_estimators":
            d_kwds_gee["numberOfTrees"] = v
        elif k == "min_samples_leaf":
            d_kwds_gee["minLeafPopulation"] = v
        elif k == "max_samples":
            v = 1 if v is None else v
            d_kwds_gee["bagFraction"] = v
        elif verbose:
            print(f"ignoring {k} to build GEE classifier")
        else:
            pass
    return d_kwds_gee


def infer_and_compute_metrics(fc, classifier, aggregate_preds=True):

    if aggregate_preds:
        classifier = classifier.setOutputMode("PROBABILITY")
        preds_proba = fc.classify(classifier)
        agg_preds = aggregate_predictions(preds_proba)
        agg_preds = agg_preds.map(lambda f: f.set("classification_bin", ee.Number(f.get("classification")).gte(0.5)))
        return compute_metrics(agg_preds, preds_name="classification_bin")
    else:
        classifier = classifier.setOutputMode("CLASSIFICATION")
        preds = fc.classify(classifier)
        return compute_metrics(preds)


def compute_metrics(preds, labels_name="label", preds_name="classification"):
    # Get confusion matrix ('classification' is the property added by RF)
    conf_matrix = preds.errorMatrix(labels_name, preds_name)  # [[tn, fp], [fn, tp]]

    # Compute metrics
    # tn = conf_matrix.array().get([0, 0])
    fp = conf_matrix.array().get([0, 1])
    fn = conf_matrix.array().get([1, 0])
    tp = conf_matrix.array().get([1, 1])

    metrics = ee.Dictionary(
        {
            "accuracy": conf_matrix.accuracy(),
            "precision": tp.divide(tp.add(fp)),
            "recall": tp.divide(tp.add(fn)),
            "f1": ee.Number(conf_matrix.fscore(1).toList().get(1)),  # F1-score for positive label
            "f05": ee.Number(conf_matrix.fscore(0.5).toList().get(1)),  # F0.5-score for positive label
        }
    )
    return metrics


def aggregate_predictions(preds):

    def aggregate_date(date):
        preds_date = preds.filter(ee.Filter.eq("start_post", date))
        unique_ids = preds_date.aggregate_array("unosat_id").distinct()

        def aggregate_id(id):
            all_preds_date_id = preds_date.filter(ee.Filter.eq("unosat_id", id))
            geo = all_preds_date_id.first().geometry()
            new_props = {
                "start_post": ee.String(all_preds_date_id.first().get("start_post")),
                "unosat_id": ee.String(id),
                "classification": all_preds_date_id.aggregate_mean("classification"),
                "label": ee.Number(all_preds_date_id.first().get("label")),
            }
            new_feature = ee.Feature(ee.Geometry(geo), new_props)
            return new_feature

        _preds = ee.FeatureCollection(unique_ids.map(aggregate_id))
        return _preds

    unique_post_dates = preds.aggregate_array("start_post").distinct()
    preds_agg = ee.FeatureCollection(unique_post_dates.map(aggregate_date)).flatten()
    return preds_agg
