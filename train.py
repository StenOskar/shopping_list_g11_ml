from services.data_loader import SupabaseDataLoader
from models.recommender import RecommendationModel

# Supabase credentials
SUPABASE_URL = "https://project.supabase.co"
SUPABASE_KEY = "api-key"

# Fetch purchase data
loader = SupabaseDataLoader(SUPABASE_URL, SUPABASE_KEY)
df = loader.fetch_purchase_history()

# Prepare data (Convert user & item names to numerical IDs)


# Train the model
