import numpy as np
import pandas as pd 


def create_data():
    np.random.seed(10)
    word_file = "/usr/share/dict/words"
    WORDS = open(word_file).read().splitlines()
    customers = pd.DataFrame({"customer_id": np.arange(1,11), 
                            "customer_age": np.random.uniform(0, 12*3, 10),
                            "customer_name": [WORDS[i] for i in np.random.randint(1, len(WORDS), 10)]})

    orders = pd.DataFrame({"order_id": np.arange(1,21), 
                        "customer_id": np.random.randint(1, 11, 20), 
                            "order_amount": np.random.uniform(0, 1000, 20),
                        })

    web_visits = pd.DataFrame({"interaction_id": np.arange(1,21), 
                            "date": pd.to_datetime(['2024-05-0' + str(int(np.floor(i/5) + 1)) for i in np.arange(1,21)]),
                            "customer_id": [None if i > 10 else i  for i in np.random.randint(1, 20, 20)] ,
                            "time_on_page": np.random.uniform(0, 100, 20),
                            "temp": np.random.randint(1, 1000, 20)
                            })
    # This is just to force the visitor_id to be the same if the customer_id is the same. 
    web_visits['visitor_id'] = np.where(~web_visits['customer_id'].isna(), web_visits['customer_id'] * 124, web_visits['temp'])

    ad_clicks = pd.DataFrame({"click_id": np.arange(1,21), 
                            "date": pd.to_datetime(['2024-05-0' + str(int(np.floor(i/3) + 3)) for i in np.arange(1,21)]),
                            "ad_id": np.random.randint(1000, 10000, 20),
                            "customer_id": [None if i > 10 else i  for i in np.random.randint(1, 20, 20)],
                            "temp": np.random.randint(1, 1000000, 20)})
    # This is just to force the external_user_id to be the same if the customer_id is the same. 
    ad_clicks['external_user_id'] = np.where(~ad_clicks['customer_id'].isna(), ad_clicks['customer_id'] * 98793, ad_clicks['temp'])

    return (customers, orders, web_visits.drop(['temp'], axis=1), ad_clicks.drop(['temp'], axis=1))