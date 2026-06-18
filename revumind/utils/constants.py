"""
Shared constants and configuration for the RevuMind project.
"""

STOP_WORDS = {
    'i','me','my','we','our','you','your','it','its','am','is','are','was',
    'were','be','been','have','has','had','do','does','did','a','an','the',
    'and','but','if','or','as','of','at','by','for','with','to','from',
    'in','on','so','too','very','just','not','also','this','that','will',
    'can','would','could','should','got','get','went','go','one','two',
    'bought','buy','product','item','order','received','delivery',
}

PALETTE = ["#5DCAA5", "#D85A30", "#EF9F27", "#7F77DD", "#378ADD",
           "#D4537E", "#97C459", "#888780"]

PLOTTING_CONFIG = {
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

def configure_plotting():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update(PLOTTING_CONFIG)
