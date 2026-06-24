def compare_models(

    production_auc,
    candidate_auc

):

    if candidate_auc > production_auc:

        return {

            "promote": True,

            "message":
                "Candidate Better"
        }

    return {

        "promote": False,

        "message":
            "Keep Production"
    }