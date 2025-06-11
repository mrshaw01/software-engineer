from collections import Counter
from random import uniform

from fuzzy_triangle_area import FuzzyTriangleArea


class MergeExperts:

    def __init__(self, mode, *experts):
        self.mode, self.experts = mode, experts

    def __call__(self, a, b, c):
        results = [exp(a, b, c) for exp in self.experts]
        if self.mode == "vote":
            return Counter(results).most_common(1)[0][0]
        elif self.mode == "mean":
            return sum(results) / len(results)


rvalues = [(uniform(0.7, 0.9), uniform(0.05, 0.2)) for _ in range(20)]
experts = [FuzzyTriangleArea(p, v) for p, v in rvalues]
merger1 = MergeExperts("vote", *experts)
merger2 = MergeExperts("mean", *experts)

print("Vote:", merger1(3, 4, 5))
print("Mean:", merger2(3, 4, 5))
