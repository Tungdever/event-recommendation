import pandas as pd

# Đọc file CSV
df = pd.read_csv('event_views.csv')

# Kiểm tra dữ liệu ban đầu
print("Before cleaning:")
print("Columns:", df.columns)
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())
print("Rating stats:\n", df['rating'].describe())

# Làm sạch dữ liệu
# 1. Giới hạn rating trong [0, 5]
df['rating'] = df['rating'].clip(lower=0, upper=5)

# 2. Ép kiểu userId và eventId thành int, xử lý giá trị không hợp lệ
try:
    df['userId'] = pd.to_numeric(df['userId'], errors='coerce').astype('Int64')
    df['eventId'] = pd.to_numeric(df['eventId'], errors='coerce').astype('Int64')
except ValueError as e:
    print(f"Error converting to int: {e}")
    df = df[df['userId'].apply(lambda x: isinstance(x, (int, float)) and x.is_integer())]
    df = df[df['eventId'].apply(lambda x: isinstance(x, (int, float)) and x.is_integer())]
    df['userId'] = df['userId'].astype('Int64')
    df['eventId'] = df['eventId'].astype('Int64')

# 3. Xóa hàng có giá trị NaN
df = df.dropna()

# Kiểm tra lại
print("\nAfter cleaning:")
print("Columns:", df.columns)
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())
print("Rating stats:\n", df['rating'].describe())
print("Unique userIds:", df['userId'].unique())
print("Unique eventIds:", df['eventId'].unique())

# Lưu file
df.to_csv('event_views_cleaned.csv', index=False)
print(f"Cleaned data saved to event_views_cleaned.csv with {len(df)} records")