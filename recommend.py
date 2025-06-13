from flask import Flask, request, jsonify
import pandas as pd
import requests
from surprise import SVD, Dataset, Reader
import logging
import schedule
import time
import threading
import random  # Thêm import random

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình API
EVENT_VIEWS_API = 'http://localhost:8080/api/events/export-event-views'
ACTIVE_EVENTS_API = 'http://localhost:8080/api/events/active-ids'

# Biến toàn cục cho mô hình và trainset
model = None
trainset = None

def load_event_views():
    """Tải và làm sạch dữ liệu từ API"""
    try:
        logger.info(f"Đang tải dữ liệu từ {EVENT_VIEWS_API}")
        response = requests.post(EVENT_VIEWS_API)
        response.raise_for_status()
        with open('event_views_temp.csv', 'w', encoding='utf-8') as f:
            f.write(response.text)
        data = pd.read_csv('event_views_temp.csv')
        logger.info(f"Đã tải {len(data)} bản ghi từ API")

        required_columns = ['userId', 'eventId', 'rating']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Thiếu cột bắt buộc: {required_columns}")
            raise ValueError(f"Dữ liệu phải chứa {required_columns}")

        # Làm sạch dữ liệu
        logger.info("Đang làm sạch dữ liệu")
        data['rating'] = data['rating'].clip(lower=0, upper=5)
        data['userId'] = pd.to_numeric(data['userId'], errors='coerce').astype('Int64')
        data['eventId'] = pd.to_numeric(data['eventId'], errors='coerce').astype('Int64')
        data = data.dropna()

        # Lọc sự kiện còn hiệu lực
        active_event_ids = get_active_event_ids()
        data = data[data['eventId'].isin(active_event_ids)]

        logger.info(f"Sau khi làm sạch: {len(data)} bản ghi")
        logger.info(f"Data types:\n{data.dtypes}")
        logger.info(f"Missing values:\n{data.isnull().sum()}")
        logger.info(f"Rating stats:\n{data['rating'].describe()}")

        if data.empty:
            logger.error("Không có dữ liệu hợp lệ sau khi làm sạch")
            raise ValueError("Không có dữ liệu hợp lệ sau khi làm sạch")

        return data
    except Exception as e:
        logger.error(f"Lỗi trong load_event_views: {str(e)}")
        return pd.DataFrame()

def get_active_event_ids():
    """Lấy danh sách eventId còn hiệu lực từ API"""
    try:
        response = requests.get(ACTIVE_EVENTS_API)
        response.raise_for_status()
        active_ids = response.json()
        logger.info(f"Đã lấy {len(active_ids)} active event IDs")
        return set(active_ids)
    except Exception as e:
        logger.error(f"Lỗi khi lấy active event IDs: {str(e)}")
        return set()

def train_model():
    """Huấn luyện mô hình SVD"""
    global model, trainset
    try:
        data = load_event_views()
        if data.empty:
            logger.warning("Không có dữ liệu để huấn luyện, sử dụng dữ liệu mặc định")
            data = pd.DataFrame({
                'userId': [1, 1, 2, 2, 20, 20, 22, 30],
                'eventId': [101, 102, 102, 103, 99, 100, 101, 99],
                'rating': [1, 2, 1, 1, 2, 1, 1, 1]
            })

        data['userId'] = data['userId'].astype(str)
        data['eventId'] = data['eventId'].astype(str)

        reader = Reader(rating_scale=(0, 5))
        dataset = Dataset.load_from_df(data[['userId', 'eventId', 'rating']], reader)
        trainset = dataset.build_full_trainset()
        logger.info(f"Danh sách userId trong trainset: {list(trainset._raw2inner_id_users.keys())}")
        logger.info(f"Danh sách eventId trong trainset: {list(trainset._raw2inner_id_items.keys())}")
        logger.info("Đang huấn luyện mô hình SVD")
        model = SVD(n_factors=30, n_epochs=20, random_state=42)
        model.fit(trainset)
        logger.info("Huấn luyện mô hình thành công")
    except Exception as e:
        logger.error(f"Lỗi trong train_model: {str(e)}")
        raise

def get_recommendations(model, trainset, user_id, all_event_ids, n=10):
    """Tạo đề xuất cho userId"""
    try:
        logger.info(f"Tạo đề xuất cho userId: {user_id}")
        user_id = str(user_id)
        if user_id not in trainset._raw2inner_id_users:
            logger.warning(f"Không tìm thấy user {user_id} trong dữ liệu huấn luyện")
            return []

        # Lấy sự kiện đã xem
        viewed_events = set()
        inner_user_id = trainset.to_inner_uid(user_id)
        for item_inner_id, _ in trainset.ur[inner_user_id]:
            raw_item_id = trainset.to_raw_iid(item_inner_id)
            viewed_events.add(raw_item_id)
        logger.info(f"Sự kiện đã xem bởi userId {user_id}: {viewed_events}")

        # Lấy sự kiện chưa xem và còn hiệu lực
        active_event_ids = get_active_event_ids()
        unviewed_events = [
            str(eid) for eid in all_event_ids
            if str(eid) not in viewed_events and str(eid) in trainset._raw2inner_id_items and eid in active_event_ids
        ]
        logger.info(f"Sự kiện chưa xem: {unviewed_events}")
        if not unviewed_events:
            logger.warning(f"Không có sự kiện chưa xem hợp lệ cho userId {user_id}")
            return []

        # Tạo dự đoán và lấy top K=30
        K = 30  # Số lượng sự kiện tiềm năng
        predictions = [model.predict(user_id, event_id).est for event_id in unviewed_events]
        top_k = sorted(zip(unviewed_events, predictions), key=lambda x: x[1], reverse=True)[:min(K, len(unviewed_events))]
        top_k_events = [event_id for event_id, _ in top_k]

        # Chọn ngẫu nhiên n=10 từ top K
        recommendations = random.sample(top_k_events, k=min(n, len(top_k_events)))
        recommendations = [int(event_id) for event_id in recommendations]
        logger.info(f"Ngẫu nhiên {n} đề xuất từ top {K}: {recommendations}")
        return recommendations
    except Exception as e:
        logger.error(f"Lỗi trong get_recommendations: {str(e)}")
        return []

# Huấn luyện mô hình lần đầu
train_model()

@app.route('/recommendations', methods=['POST'])
def recommend():
    try:
        if not request.is_json:
            logger.error("Yêu cầu phải là JSON")
            return jsonify({"error": "Content-Type phải là application/json"}), 415
        data = request.json
        logger.info(f"Nhận được yêu cầu: {data}")
        user_id = data.get('userId')
        all_event_ids = data.get('allEventIds', [])
        if user_id is None or not isinstance(user_id, int):
            logger.error(f"userId không hợp lệ: {user_id}")
            return jsonify({"error": "userId phải là số nguyên"}), 400
        if not isinstance(all_event_ids, list) or not all(isinstance(eid, int) for eid in all_event_ids):
            logger.error(f"allEventIds không hợp lệ: {all_event_ids}")
            return jsonify({"error": "allEventIds phải là danh sách số nguyên"}), 400
        recommendations = get_recommendations(model, trainset, user_id, all_event_ids)
        return jsonify({'eventIds': recommendations})
    except Exception as e:
        logger.error(f"Lỗi trong recommend: {str(e)}")
        return jsonify({"error": "Lỗi máy chủ nội bộ"}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        logger.info("Đang huấn luyện lại mô hình")
        global model, trainset
        train_model()
        return jsonify({'status': 'Mô hình đã được huấn luyện lại thành công'})
    except Exception as e:
        logger.error(f"Lỗi trong retrain: {str(e)}")
        return jsonify({"error": "Lỗi máy chủ nội bộ"}), 500

# Scheduler để tái huấn luyện mô hình
def run_scheduler():
    schedule.every().day.at("02:00").do(train_model)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Khởi động scheduler trong luồng riêng
threading.Thread(target=run_scheduler, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)