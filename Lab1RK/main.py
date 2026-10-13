import requests
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Запит до API (Open-Meteo)
# ==========================================
print("--- Крок 1-4: Отримання та табуляція даних ---")

lats = [48.164214, 48.164983, 48.165605, 48.166228, 48.166777, 48.167326, 48.167011, 48.166053, 48.166655, 48.166497, 48.166128, 48.165416, 48.164546, 48.163412, 48.162331, 48.162015, 48.162147, 48.161751, 48.161197, 48.160580, 48.160250]
lons = [24.536044, 24.534836, 24.534068, 24.532915, 24.531927, 24.530884, 24.530061, 24.528039, 24.526064, 24.523574, 24.520214, 24.517170, 24.514640, 24.512980, 24.511715, 24.509462, 24.506932, 24.504244, 24.501793, 24.500537, 24.500106]

url = f"https://api.open-meteo.com/v1/elevation?latitude={','.join(map(str, lats))}&longitude={','.join(map(str, lons))}"

response = requests.get(url)
data = response.json()

# Пакуємо дані у формат (як в Open-Elevation)
results = []
for i in range(len(lats)):
    results.append({
        "latitude": lats[i],
        "longitude": lons[i],
        "elevation": data['elevation'][i]
    })

n = len(results)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

print(f"Отримано вузлів: {n}")

# ==========================================
# Функції для побудови кубічного сплайна
# ==========================================
def build_cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

    # Ініціалізація векторів для трьохдіагональної матриці
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    F = np.zeros(n)

    # Формування системи рівнянь для внутрішніх вузлів (вільні сплайни)
    B[0] = 1.0
    B[-1] = 1.0
    for i in range(1, n - 1):
        A[i] = h[i - 1]
        B[i] = 2.0 * (h[i - 1] + h[i])
        C[i] = h[i]
        F[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # МЕТОД ПРОГОНКИ (Прямий хід)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    for i in range(1, n):
        m = B[i] - A[i] * alpha[i - 1]
        alpha[i] = C[i] / m
        beta[i] = (F[i] - A[i] * beta[i - 1]) / m

    # МЕТОД ПРОГОНКИ (Зворотний хід - знаходимо коефіцієнти 'c')
    c = np.zeros(n)
    for i in range(n - 2, -1, -1):
        c[i] = beta[i] - alpha[i] * c[i + 1]

    # Обчислення коефіцієнтів a, b, d
    a = y[:-1]
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    return a, b, c[:-1], d


def evaluate_spline(x_eval, x, a, b, c, d):
    y_eval = np.zeros_like(x_eval)
    for j, xv in enumerate(x_eval):
        # Пошук інтервалу
        idx = np.searchsorted(x, xv) - 1
        idx = max(0, min(idx, len(x) - 2))
        dx = xv - x[idx]
        y_eval[j] = a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3
    return y_eval


# ==========================================
# Обчислення коефіцієнтів для повного набору
# ==========================================
print("\n--- Крок 9: Обчислення коефіцієнтів сплайна (повний набір) ---")
a, b, c, d = build_cubic_spline(distances, elevations)
print(f"{'Інтервал':<10} | {'a':<8} | {'b':<10} | {'c':<10} | {'d':<10}")
for i in range(3):  # Виводимо перші 3 для прикладу, щоб не засмічувати консоль
    print(f"[{i:2d} - {i + 1:2d}] | {a[i]:8.2f} | {b[i]:10.4f} | {c[i]:10.4f} | {d[i]:10.6f}")
print("... (інші коефіцієнти обчислено) ...")

# ==========================================
# Побудова графіків (10, 15, повна к-сть вузлів)
# ==========================================
plt.figure(figsize=(14, 8))
x_dense = np.linspace(min(distances), max(distances), 500)

# 1. Повний набір (Reference)
y_dense_full = evaluate_spline(x_dense, distances, a, b, c, d)
plt.plot(x_dense, y_dense_full, 'k-', linewidth=2, label='Еталонний сплайн (21 вузол)')

# 2. Набір 10 вузлів
idx_10 = np.linspace(0, n - 1, 10, dtype=int)
dist_10, elev_10 = np.array(distances)[idx_10], np.array(elevations)[idx_10]
a10, b10, c10, d10 = build_cubic_spline(dist_10, elev_10)
y_dense_10 = evaluate_spline(x_dense, dist_10, a10, b10, c10, d10)
plt.plot(x_dense, y_dense_10, 'r--', label='Сплайн (10 вузлів)')
plt.scatter(dist_10, elev_10, color='red', zorder=5)

# 3. Набір 15 вузлів
idx_15 = np.linspace(0, n - 1, 15, dtype=int)
dist_15, elev_15 = np.array(distances)[idx_15], np.array(elevations)[idx_15]
a15, b15, c15, d15 = build_cubic_spline(dist_15, elev_15)
y_dense_15 = evaluate_spline(x_dense, dist_15, a15, b15, c15, d15)
plt.plot(x_dense, y_dense_15, 'g-.', label='Сплайн (15 вузлів)')
plt.scatter(dist_15, elev_15, color='green', zorder=5)

plt.title("Профіль висоти маршруту: Заросляк - Говерла (Інтерполяція)")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# Характеристики маршруту
# ==========================================
print("\n--- Додатково: Аналіз маршруту ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")

total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
print(f"Сумарний спуск (м): {total_descent:.2f}")

# Градієнт
grad_full = np.gradient(y_dense_full, x_dense) * 100
print(f"Максимальний підйом (% крутизни): {np.max(grad_full):.2f}%")
print(f"Максимальний спуск (% крутизни): {np.min(grad_full):.2f}%")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}%")

# Механічна енергія
mass = 80
g = 9.81
energy = mass * g * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Витрачена енергія (ккал): {energy / 4184:.2f}")

# Запис у файл
n = len(results)

print("\nТабуляція вузлів (запис у файл):")

with open("tabulation_results.txt", "w", encoding="utf-8") as file:
    file.write(f"Кількість вузлів: {n}\n")
    file.write("№  | Latitude  | Longitude | Elevation (m)\n")
    file.write("-" * 46 + "\n")

    for i, point in enumerate(results):
        line = f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}"
        file.write(line + "\n")

print("\n✅ Дані успішно збережено у файл 'tabulation_results.txt'")