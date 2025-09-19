import numpy as np
from collections import defaultdict

# сопоставление аминокислот числам
aa_map = defaultdict(int)
alphabet = "ACDEFGHIKLMNPQRSTVWY"
# нумеруем символы алфавита
for i, aa in enumerate(alphabet, 1):
    aa_map[aa] = i


# чтение fasta с сохранением заголовка и последовательности
def read_fasta_gz(path):
    with open(path, "r") as f:
        header = None
        seq = []
        # Читаем файл построчно
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Если строка начинается с ">", это заголовок новой записи
            if line.startswith(">"):
                # Если уже есть предыдущий заголовок и накопленная последовательность —
                # возвращаем их как готовую запись (yield)
                if header and seq:
                    yield header, "".join(seq)
                # Начинаем новую запись: сохраняем заголовок и сбрасываем накопитель последовательности
                header = line
                seq = []
            else:
                # Иначе — это часть последовательности: добавляем строку в накопитель
                seq.append(line)
        # После окончания файла: если осталась последняя запись — возвращаем и её
        if header and seq:
            yield header, "".join(seq)


def seq_to_signal(seq):
    # Преобразуем последовательность в числовой сигнал. Напрмиер, "MAF" → [M=11, A=1, F=6] → [11, 1, 6]
    signal = np.array([aa_map[aa] for aa in seq if aa in aa_map], dtype=float)
    # Находим евклидову норму получившегося вектора
    norm = np.linalg.norm(signal)
    if norm > 0:
        # Нормируем вектор, используя евклидову норму
        signal = signal / norm
    return signal


def energy_metric(seq1, seq2):
    # Переводим обе строки в числовые вектора
    s1 = seq_to_signal(seq1)
    s2 = seq_to_signal(seq2)

    n = max(len(s1), len(s2))
    # Дополняем нулями до одинаковой длины, чтобы можно было поэлементно сравнивать векторы одинаковой длины
    s1 = np.pad(s1, (0, n - len(s1)), mode='constant')
    s2 = np.pad(s2, (0, n - len(s2)), mode='constant')

    # Возвращаем расстояние между двумя векторами в пространстве (евклидову норму между ними)
    return np.linalg.norm(s1 - s2)


def find_similar(user_seq, input_path, output_path):
    # Список для хранения всех вычисленных записей
    metrics = []
    # Читаем все записи из FASTA-файла в список
    records = list(read_fasta_gz(input_path))

    # Проходим по всем записям с индексом (i) и распаковываем (header, sequence)
    for i, (header, seq) in enumerate(records):
        # Вычисляем метрику схожести между пользовательской последовательностью и текущей
        val = energy_metric(user_seq, seq)
        # вытащим название (например, "Frog virus ...")
        description = " ".join(header.split()[1:]) if len(header.split()) > 1 else ""
        metrics.append((i, val, seq, description))

    # Добавляем кортеж с данными в список метрик
    # Структура: (порядковый_индекс_в_файле, значение_метрики, сама_последовательность, описание_из_заголовка)
    metrics_sorted = sorted(metrics, key=lambda x: x[1])

    with open(output_path, "w") as f:
        for idx, val, seq, descr in metrics_sorted:
            # Записываем строку с информацией: индекс, метрика (6 знаков после запятой), длина, описание
            f.write(f"Индекс: {idx}, Метрика: {val:.6f}, Длина: {len(seq)}, Описание: {descr}\n")
            # Записываем саму последовательность и добавляем пустую строку для разделения записей
            f.write(seq + "\n\n")

    print("Результат сохранён в", output_path)


def main():
    # путь к файлу с данными
    path = "uniprot_sprot.fasta"
    
    # сюда будут записываться выходные данные
    output_path = "sorted_by_user_sequence.txt"
    
    # Пользователь вводит нужную последовательность для проверки
    user_seq = input()

    # Запуск алгоритма для строки, введенной пользователем
    find_similar(user_seq, path, output_path)


if __name__ == "__main__":
    main()
