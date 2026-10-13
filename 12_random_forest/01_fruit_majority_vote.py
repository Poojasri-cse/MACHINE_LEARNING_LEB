from collections import Counter
import matplotlib.pyplot as plt

print("Worksheet 12.6 - Exercise 1: Fruit basket majority voting")
tree_outputs = {
    "Tree-1": "Apple (Class A)",
    "Tree-2": "Apple (Class A)",
    "Tree-3": "Banana (Class B)"
}
votes = Counter(tree_outputs.values())
print("Votes from individual trees:")
for tree, result in tree_outputs.items():
    print(f"{tree} -> {result}")
print("Vote count:", dict(votes))
winner, count = votes.most_common(1)[0]
print(f"Majority voting -> Final class: {winner} ({count} of {len(tree_outputs)} trees)")
plt.bar(votes.keys(), votes.values())
plt.ylabel("Number of trees")
plt.title("Majority voting in a random forest")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()
