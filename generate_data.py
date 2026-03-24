import pandas as pd
import random

def generate_library_data(n_rows=500):
    subjects = ["Computer Science", "Mathematics", "Physics", "Management", "Biology", "Economics"]
    data = []

    for i in range(1, n_rows + 1):
        subject = random.choice(subjects)
        semester = random.randint(1, 8)
        past_borrow_count = random.randint(0, 200)
        course_relevance = random.choice([0, 1])

        if past_borrow_count > 120 and course_relevance == 1:
            demand_level = "High"
        elif past_borrow_count > 40:
            demand_level = "Medium"
        else:
            demand_level = "Low"

        data.append([
            i,
            subject,
            semester,
            past_borrow_count,
            course_relevance,
            demand_level
        ])

    columns = ["book_id", "subject", "semester", "past_borrow_count", "course_relevance", "demand_level"]
    df = pd.DataFrame(data, columns=columns)
    return df


if __name__ == "__main__":
    df = generate_library_data(500)
    df.to_csv("library_data.csv", index=False)
    print("library_data.csv generated successfully!")
