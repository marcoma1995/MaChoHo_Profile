import numpy as np

import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj

# Create mothers involvement plot.
# width of the bars.
barWidth = 0.5

# Choose the height of the mixed mothers involvment bars.
bars1 = [0.09, 0.09, 0.04, 0.05, 0.15, 0.04]

# Choose the height of the female mothers involvement bar.
bars2 = [0.13, 0.07, 0.02, 0.03, 0.13, 0.06]

# Choose the height of the male mothers involvement bar.
bars3 = [0.06, 0.06, 0.01, 0.04, 0.11, -0.02]

# Choose the height of the error bars mother both.
yer1 = [0.04, 0.03, 0.03, 0.03, 0.03, 0.03]

# Choose the height of the error bars mother female.
yer2 = [0.05, 0.05, 0.04, 0.05, 0.04, 0.04]

# Choose the height of the error bars mother male.
yer3 = [0.05, 0.04, 0.04, 0.05, 0.04, 0.05]

# The x position of bars.
r1 = (np.arange(len(bars1))) * 1.7
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create blue bars.
plt.bar(r1, bars1, width=barWidth, color='blue', edgecolor='black',
        ecolor='grey', yerr=yer1, capsize=3, label='mixed sample')

# Create cyan bars.
plt.bar(r2, bars2, width=barWidth, color='red', edgecolor='black',
        ecolor='grey', yerr=yer2, capsize=3, label='only females')

# Create cyan bars.
plt.bar(r3, bars3, width=barWidth, color='green', edgecolor='black',
        ecolor='grey', yerr=yer3, capsize=3, label='only males')

# General layout.
plt.xticks(r2, ['Locus of control', 'Openness to experience',
                'Conscientiousness', 'Extraversion', 'Agreeableness',
                'Neuroticism'], rotation=90)
plt.yticks(np.arange(-0.20, 0.30, 0.05))
plt.ylim(-0.25, 0.30)

# Adjust top and bottom.
plt.subplots_adjust(bottom=0.50, top=0.93)
plt.title("Figure 1: Mother's involvement parameter compared among samples")
plt.xlabel('')
plt.ylabel('Parameter intensity')
plt.legend()

# Save as jpeg.
plt.savefig(ppj("OUT_FIGURES", "figure_maternal_involvement_personality.jpeg"))

# Create fathers involvement plot.
# width of the bars.
barWidth = 0.5

# Choose the height of the mixed fathers involvment bars.
bars4 = [0.08, 0.04, 0.16, 0.09, 0.03, -0.10]

# Choose the height of the female fathers involvement bar.
bars5 = [0.02, 0.10, 0.16, 0.06, 0.04, -0.17]

# Choose the height of the male fathers involvement bar.
bars6 = [0.14, 0.04, 0.22, 0.17, 0.09, 0.01]

# Choose the height of the error bars father both.
yer4 = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04]

# Choose the height of the error bars father female.
yer5 = [0.05, 0.06, 0.05, 0.06, 0.05, 0.05]

# Choose the height of the error bars father male.
yer6 = [0.06, 0.05, 0.05, 0.06, 0.05, 0.06]

# The x position of bars.
r4 = (np.arange(len(bars4))) * 1.7
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]


# General layout.
plt.xticks(r5, ['Locus of control', 'Openness to experience',
                'Conscientiousness', 'Extraversion', 'Agreeableness',
                'Neuroticism'], rotation=90)
plt.yticks(np.arange(-0.20, 0.30, 0.05))
plt.ylim(-0.23, 0.30)

# Add title and axis names.
plt.subplots_adjust(bottom=0.50, top=0.93)
plt.title("Figure 2: Father's involvement parameter compared among samples")
plt.xlabel('')
plt.ylabel('Parameter intensity')
plt.legend()

# Save as jpeg.
plt.savefig(ppj("OUT_FIGURES", "figure_paternal_involvement_personality.jpeg"))


# Create difference between mother's and father's involvement.
# Plot father - mother
# width of the bars
barWidth = 0.5

# Choose the height of the mixed fathers involvment bars.
bars7 = [a - b for a, b in zip(bars4, bars1)]

# Choose the height of the female fathers involvement bar.
bars8 = [a - b for a, b in zip(bars5, bars2)]

# Choose the height of the male fathers involvement bar.
bars9 = [a - b for a, b in zip(bars6, bars3)]

# The x position of bars.
r7 = (np.arange(len(bars7))) * 1.7
r8 = [x + barWidth for x in r7]
r9 = [x + barWidth for x in r8]

# General layout.
plt.xticks(r8, ['Locus of control', 'Openness to experience',
                'Conscientiousness', 'Extraversion', 'Agreeableness',
                'Neuroticism'], rotation=90)
plt.yticks(np.arange(-0.25, 0.30, 0.05))
plt.ylim(-0.25, 0.27)

# Add title and axis names.
plt.subplots_adjust(bottom=0.50, top=0.93)
plt.title("Figure 3: Difference in involvement parameters (father - mother)")
plt.xlabel('')
plt.ylabel('Magnitude')
plt.legend()

# Save as jpeg.
plt.savefig(ppj("OUT_FIGURES", "figure_parental_involvement_difference.jpeg"))


# Create combined involvement  plot father + mother
# width of the bars
barWidth = 0.5

# Choose the height of the mixed fathers involvment bars.
bars7 = [a + b for a, b in zip(bars4, bars1)]

# Choose the height of the female fathers involvement bar.
bars8 = [a + b for a, b in zip(bars5, bars2)]

# Choose the height of the male fathers involvement bar.
bars9 = [a + b for a, b in zip(bars6, bars3)]

# The x position of bars/
r7 = (np.arange(len(bars7))) * 1.7
r8 = [x + barWidth for x in r7]
r9 = [x + barWidth for x in r8]


# General layout.
plt.xticks(r8, ['Locus of control', 'Openness to experience',
                'Conscientiousness', 'Extraversion', 'Agreeableness',
                'Neuroticism'], rotation=90)
plt.yticks(np.arange(-0.25, 0.30, 0.05))
plt.ylim(-0.25, 0.27)

# Add title and axis names.
plt.subplots_adjust(bottom=0.50, top=0.93)
plt.title("Figure 4: Combined involvement parameters (father + mother)")
plt.xlabel('')
plt.ylabel('Magnitude')
plt.legend()

# Save as jpeg.
plt.savefig(ppj("OUT_FIGURES", "figure_parental_involvement_added_up.jpeg"))
