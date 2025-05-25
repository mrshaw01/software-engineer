"""Exercise: Dataclass for book information."""

from dataclasses import dataclass


@dataclass
class Book:
    title: str
    author: str
    isbn: str
    publication_year: int
    genre: str


book1 = Book("The Great Gatsby", "F. Scott Fitzgerald", "9780743273565", 1925, "Fiction")
book2 = Book("To Kill a Mockingbird", "Harper Lee", "9780061120084", 1960, "Fiction")
book3 = Book("1984", "George Orwell", "9780451524935", 1949, "Science Fiction")

for i, book in enumerate([book1, book2, book3], 1):
    print(f"Book {i}:")
    print(f"Title: {book.title}")
    print(f"Author: {book.author}")
    print(f"ISBN: {book.isbn}")
    print(f"Publication Year: {book.publication_year}")
    print(f"Genre: {book.genre}\n")
