import os
import pickle as pkl
import skeleton_utils
import numpy as np
import math
import scipy
import plot_utils
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from visualizeReconstructedSkeleton import reconstructSequenceSingleFrame
import pa_utils

upper_body_index=[0,11,12,13,14,15,16,23,24,33,34]
lower_body_index=[23,24,25,26,27,28,29,30,31,33]

hip_index=33
head_index=0

output_root="thesis_plots"


def total_body_MPJPE(dataset,save=True):
    framerate=21
    for e in ["squat","plank"]:
        with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{e}_good/rep_1.pkl", 'rb') as file:
            golden_reference = np.array(pkl.load(file))
        #computeSequenceSimilarityScore(reference_sequence=golden_reference, testing_sequence=golden_reference)
        exercises=os.listdir(f"../data/pkl/{dataset}/single_repetitions_ok/21_frames")
        exercises_variant=[variant for variant in exercises if e in variant]
        exercise_similarities=[]
        for exercise in exercises_variant:
            columns = []
            print(exercise)
            reps_similarities=[]
            for i in range(2,9):
                rep_name=f"rep {i}"
                print("\t",rep_name)
                with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{exercise}/rep_{i}.pkl", 'rb') as file:
                    good_repetition = np.array(pkl.load(file))
                score=pa_utils.computeSequenceSimilarityScore_v1(reference_sequence=golden_reference, testing_sequence=good_repetition,hip_index=hip_index,head_index=head_index)
                reps_similarities.append(score)
                columns.append(rep_name)
            print("="*85)
            exercise_similarities.append(reps_similarities)
        similarity_matrix=np.array(exercise_similarities)
        exercises_variant_display = [name.replace("butt", "hips") for name in exercises_variant]
        output_path = None
        if save:
            output_path = os.path.join(output_root, dataset, f"{e}_1.png")
        pa_utils.createHeatmap(similarity_matrix=similarity_matrix, path=output_path, columns=columns,
                               rows=exercises_variant_display)
        #computeSequenceSimilarityScore(reference_sequence=golden_reference, testing_sequence=bad_repetition)
    #######################LUNGE################
    e = "lunge"
    with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{e}_good/rep_0.pkl", 'rb') as file:
        golden_reference1 = np.array(pkl.load(file))
    with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{e}_good/rep_5.pkl", 'rb') as file:
        golden_reference2 = np.array(pkl.load(file))
    exercises = os.listdir(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames")
    lunge_variant = [variant for variant in exercises if e in variant]
    lunge_exercise_similarities = []
    for exercise in lunge_variant:
        columns = []
        print(exercise)
        reps_similarities = []
        for i in range(2, 9):
            if i != 5:
                rep_name = f"rep {i}"
                print("\t", rep_name)
                with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{exercise}/rep_{i}.pkl",
                          'rb') as file:
                    test_repetition = np.array(pkl.load(file))
                score1 = pa_utils.computeSequenceSimilarityScore_v1(reference_sequence=golden_reference1,
                                                                    testing_sequence=test_repetition,
                                                                    hip_index=hip_index,head_index=head_index)
                score2 = pa_utils.computeSequenceSimilarityScore_v1(reference_sequence=golden_reference2,
                                                                    testing_sequence=test_repetition,
                                                                    hip_index=hip_index,head_index=head_index)
                score = max(score1, score2)
                reps_similarities.append(score)
                columns.append(rep_name)
        print("=" * 85)
        lunge_exercise_similarities.append(reps_similarities)
    lunge_similarity_matrix = np.array(lunge_exercise_similarities)
    output_path = None
    if save:
        output_path = os.path.join(output_root, dataset, f"{e}_1.png")
    pa_utils.createHeatmap(similarity_matrix=lunge_similarity_matrix, path=output_path, columns=columns,
                           rows=lunge_variant)


def lower_and_upper_body_MPJPE(dataset,save=True):
    framerate = 21
    exercises = os.listdir(f"../data/pkl/{dataset}/single_repetitions_ok/21_frames")
    for e in ["squat","plank"]:
        with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{e}_good/rep_1.pkl", 'rb') as file:
            golden_reference = np.array(pkl.load(file))
        # computeSequenceSimilarityScore(reference_sequence=golden_reference, testing_sequence=golden_reference)
        exercises_variant = [variant for variant in exercises if e in variant]
        exercise_low_similarities = []
        exercise_up_similarities=[]
        for exercise in exercises_variant:
            columns = []
            print(exercise)
            up_reps_similarities = []
            low_reps_similarities=[]
            for i in range(2, 9):
                rep_name = f"rep {i}"
                print("\t", rep_name)
                with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{exercise}/rep_{i}.pkl", 'rb') as file:
                    test_repetition = np.array(pkl.load(file))
                upper_score,lower_score = pa_utils.computeSequenceSimilarityScore_v2(reference_sequence=golden_reference,
                                                                            testing_sequence=test_repetition,upper_joints_index=upper_body_index,lower_joints_index=lower_body_index)
                up_reps_similarities.append(upper_score)
                low_reps_similarities.append(lower_score)
                columns.append(rep_name)
            print("=" * 85)
            exercise_up_similarities.append(up_reps_similarities)
            exercise_low_similarities.append(low_reps_similarities)
        up_similarity_matrix = np.array(exercise_up_similarities)
        low_similarity_matrix=np.array(exercise_low_similarities)
        exercises_variant_display = [name.replace("butt", "hips") for name in exercises_variant]
        if save:
            output_path_up = os.path.join(output_root, dataset, f"{e}_2_upper.png")
            output_path_low = os.path.join(output_root, dataset, f"{e}_2_lower.png")
            pa_utils.createHeatmap(similarity_matrix=up_similarity_matrix, path=output_path_up, columns=columns,
                                   rows=exercises_variant_display, title="Upper body positional similarity score")
            pa_utils.createHeatmap(similarity_matrix=low_similarity_matrix, path=output_path_low, columns=columns,
                                   rows=exercises_variant_display, title="Lower body positional similarity score")
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
            sns.heatmap(up_similarity_matrix, annot=True, cmap='YlGnBu', cbar=True, xticklabels=columns,
                        yticklabels=exercises_variant_display, ax=axes[0])
            axes[0].set_title('Upper Body')
            sns.heatmap(low_similarity_matrix, annot=True, cmap='YlGnBu', cbar=True, xticklabels=columns,
                        yticklabels=exercises_variant_display, ax=axes[1])
            axes[1].set_title('Lower Body')
            plt.suptitle("Score= ln(1/MPJPE)")
            plt.tight_layout()
            plt.show()
    #######################LUNGE################
    e="lunge"
    with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{e}_good/rep_0.pkl", 'rb') as file:
        golden_reference1 = np.array(pkl.load(file))
    with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{e}_good/rep_5.pkl", 'rb') as file:
        golden_reference2 = np.array(pkl.load(file))
        # computeSequenceSimilarityScore(reference_sequence=golden_reference, testing_sequence=golden_reference)
    exercises = os.listdir(f"../data/pkl/{dataset}/single_repetitions_ok/21_frames")
    lunge_variant = [variant for variant in exercises if e in variant]
    exercise_low_similarities = []
    exercise_up_similarities=[]
    for exercise in lunge_variant:
        columns = []
        print(exercise)
        up_reps_similarities = []
        low_reps_similarities=[]
        for i in range(2, 9):
            if i!=5:
                rep_name = f"rep {i}"
                print("\t", rep_name)
                with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{exercise}/rep_{i}.pkl", 'rb') as file:
                    test_repetition = np.array(pkl.load(file))
                upper_score1,lower_score1 = pa_utils.computeSequenceSimilarityScore_v2(reference_sequence=golden_reference1,
                                                                              testing_sequence=test_repetition,upper_joints_index=upper_body_index,lower_joints_index=lower_body_index)
                upper_score2, lower_score2 = pa_utils.computeSequenceSimilarityScore_v2(reference_sequence=golden_reference2,
                                                                               testing_sequence=test_repetition,upper_joints_index=upper_body_index,lower_joints_index=lower_body_index)
                upper_score=max(upper_score1,upper_score2)
                lower_score = max(lower_score1, lower_score2)
                up_reps_similarities.append(upper_score)
                low_reps_similarities.append(lower_score)
                columns.append(rep_name)
        print("=" * 85)
        exercise_up_similarities.append(up_reps_similarities)
        exercise_low_similarities.append(low_reps_similarities)
    up_similarity_matrix = np.array(exercise_up_similarities)
    low_similarity_matrix=np.array(exercise_low_similarities)
    if save:
        output_path_up = os.path.join(output_root, dataset, f"{e}_2_upper.png")
        output_path_low = os.path.join(output_root, dataset, f"{e}_2_lower.png")
        pa_utils.createHeatmap(similarity_matrix=up_similarity_matrix, path=output_path_up, columns=columns,
                               rows=lunge_variant, title="Upper body positional similarity score")
        pa_utils.createHeatmap(similarity_matrix=low_similarity_matrix, path=output_path_low, columns=columns,
                               rows=lunge_variant, title="Lower body positional similarity score")
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        sns.heatmap(up_similarity_matrix, annot=True, cmap='YlGnBu', cbar=True, xticklabels=columns,
                    yticklabels=exercises_variant_display, ax=axes[0])
        axes[0].set_title('Upper Body')
        sns.heatmap(low_similarity_matrix, annot=True, cmap='YlGnBu', cbar=True, xticklabels=columns,
                    yticklabels=exercises_variant_display, ax=axes[1])
        axes[1].set_title('Lower Body')
        plt.suptitle("Score= ln(1/MPJPE)")
        plt.tight_layout()
        plt.show()


def theta_comparison(dataset,save=True):
    framerate = 21
    for e in ["squat","plank","lunge"]:
        with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{e}_good/rep_1.pkl", 'rb') as file:
            golden_reference = np.array(pkl.load(file))
        # computeSequenceSimilarityScore(reference_sequence=golden_reference, testing_sequence=golden_reference)
        exercises = os.listdir(f"../data/pkl/{dataset}/single_repetitions_ok/21_frames")
        exercises_variant = [variant for variant in exercises if e in variant]
        exercise_similarities = []
        for exercise in exercises_variant:
            columns = []
            print(exercise)
            reps_similarities = []
            for i in range(2, 9):
                rep_name = f"rep {i}"
                print("\t", rep_name)
                with open(f"../data/pkl/{dataset}/single_repetitions_ok/{framerate}_frames/{exercise}/rep_{i}.pkl", 'rb') as file:
                    good_repetition = np.array(pkl.load(file))
                score = pa_utils.computeSequenceSimilarityScore_v3(reference_sequence=golden_reference,
                                                          testing_sequence=good_repetition,hip_index=33,chest_index=34)

                reps_similarities.append(score)
                columns.append(rep_name)
            print("=" * 85)
            exercise_similarities.append(reps_similarities)
        similarity_matrix = np.array(exercise_similarities)
        exercises_variant_display = [name.replace("butt", "hips") for name in exercises_variant]
        output_path = None
        if save:
            output_path = os.path.join(output_root, dataset, f"{e}_3.png")
        pa_utils.createHeatmap(similarity_matrix=similarity_matrix, cmap="coolwarm", columns=columns,
                               rows=exercises_variant_display, path=output_path, title=r"Back Angle Displacement")


if __name__ == '__main__':
    dataset="rgb"
    if not os.path.exists(os.path.join(output_root,dataset)):
        os.makedirs(os.path.join(output_root,dataset))
    total_body_MPJPE(dataset)
    lower_and_upper_body_MPJPE(dataset)
    theta_comparison(dataset)