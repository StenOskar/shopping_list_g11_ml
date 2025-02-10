from supabase import create_client
import pandas as pd


# SupabaseDataLoader class to fetch data from Supabase
class SupabaseDataLoader:
    def __init__(self, url, key):
        self.supabase = create_client(url, key)

    def fetch_purchase_history(self):
        response = self.supabase.table('purchase_history').select('*').execute()
        return pd.DataFrame(response.data)
