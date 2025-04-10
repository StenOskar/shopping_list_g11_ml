from supabase import create_client
import pandas as pd


# SupabaseDataLoader class to fetch data from Supabase
class SupabaseDataLoader:
    def __init__(self, url, key):
        self.supabase = create_client(url, key)

    def fetch_receipts(self):
        response = self.supabase.table('receipts').select('*').execute()
        return pd.DataFrame(response.data)

    def fetch_receipt_items(self):
        response = self.supabase.table('receipt_items').select('*').execute()
        return pd.DataFrame(response.data)
