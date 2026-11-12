# -*- coding: cp949 -*-
import librosa
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import scipy
import torchaudio

def compute_spectral_flatness(y, n_fft=1024, hop_length=512):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S)
    flatness = librosa.feature.spectral_flatness(S=S_mag)
    return float(np.mean(flatness))

def compute_rms_energy(y, frame_length=1024, hop_length=512):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    return float(np.mean(rms))


def compute_spectral_entropy(y, n_fft=1024, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    ps = S / np.sum(S, axis=0, keepdims=True)  # probability of each frame
    entropy = scipy.stats.entropy(ps, axis=0, base=2)
    max_entropy = np.log2(ps.shape[0])  # log2(N), where N = frequency bins
    normalized_entropy = entropy / max_entropy  # normalized spectral entropy [0,1]
    return float(np.mean(normalized_entropy))        

'''
def compute_quality_score(flatness, rms, rms_min=0.001, rms_max=0.8, alpha=0.7, beta=0.3): # spectral flatness + rms
    # RMS normalization
    rms_score = (rms - rms_min) / (rms_max - rms_min)
    rms_score = np.clip(rms, 0.0, 1.0)
    
    # quality_score calculation
    raw_score = -alpha * flatness + beta * rms_score
    
    # Min-max normalization
    # flatness \in [0, 1], rms_score \in [0, 1]:
    # min = -alpha * 1 + beta * 0 = -alpha
    # max = -alpha * 0 + beta * 1 = beta
    score_min = -alpha
    score_max = beta

    # min-max scaling to [0, 1]
    norm_score = (raw_score - score_min) / (score_max - score_min)
    norm_score = np.clip(norm_score, 0.0, 1.0)
    
    return float(norm_score)
'''

def compute_quality_score_entropy_rms(entropy_norm, rms, rms_min=0.001, rms_max=0.8, alpha=0.7, beta=0.3): # spectral entropy + rms
    """
    entropy_norm: normalized spectral entropy (0~1, higher = more noise-like)
    rms: RMS energy
    alpha: weight for entropy (noise penalty), higher alpha = more sensitive to noise
    beta: weight for rms (signal strength reward)

    Returns:
        quality score ∈ [0, 1] (higher = better quality)
    """

    # Normalize RMS
    rms_score = (rms - rms_min) / (rms_max - rms_min)
    rms_score = np.clip(rms_score, 0.0, 1.0)

    # Quality score: noise penalty (entropy), signal reward (rms)
    # higher entropy = more noise-like -> low quality
    raw_score = -alpha * entropy_norm + beta * rms_score

    # min-max normalization
    score_min = -alpha   # entropy=1, rms=0
    score_max = beta     # entropy=0, rms=1

    norm_score = (raw_score - score_min) / (score_max - score_min)
    norm_score = np.clip(norm_score, 0.0, 1.0)

    return float(norm_score)
    
    
    
if __name__ == '__main__':
    #data_all = sorted(glob('/home2/jw/workspace/mcl/multitask_embc/data/snubh_dataset/data/*.wav'))
    data_all = sorted(glob('/home2/jw/workspace/mcl/multitask_embc/data/icbhi_wav/training/*.wav'))
    
    
    #flatness_list = []
    entropy_list = []
    rms_list = []
    file_list = []
    quality_list = []
    quality_list2 = []
    
    for i, data in enumerate(data_all):
        y, sr = librosa.load(data, sr=None)        
        '''
        y, sr = torchaudio.load(data)        
        y = y.numpy().squeeze(0)
        '''
                
        #flatness = compute_spectral_flatness(y, sr)
        entropy = compute_spectral_entropy(y)
        rms = compute_rms_energy(y)
        
        quality_score = compute_quality_score_entropy_rms(entropy, rms)
        quality_score2 = compute_quality_score_entropy_rms(entropy, rms, alpha=0.3, beta=0.7)
        
        
        #flatness_list.append(flatness)
        entropy_list.append(entropy)
        rms_list.append(rms)
        quality_list.append(quality_score)
        quality_list2.append(quality_score2)
        
        file_list.append(data)
    
    df = pd.DataFrame({
        'filename': file_list,
        #'flatness': flatness_list,
        'entropy': entropy_list,
        'rms': rms_list,
        'quality_score': quality_list,
        'quality_score2': quality_list2
    })
    
    df.to_csv('icbhi_training_entropy_rms_distribution.csv', index=False)
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Flatness Histogram
    plt.figure()
    #sns.histplot(df['flatness'], bins=30, kde=True)
    sns.histplot(df['entropy'], bins=30, kde=True)
    plt.title("Spectral Entropy Distribution of ICBHI Training Samples")
    plt.xlabel("Spectral Entropy")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig('./icbhi_training_spectral_entropy.png')
    plt.show()
    
    # RMS Histogram
    plt.figure()
    sns.histplot(df['rms'], bins=30, kde=True)
    plt.title("RMS Energy Distribution of ICBHI Training Samples")
    plt.xlabel("RMS")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig('./icbhi_training_rms.png')
    plt.show()
    
    
    # Quality_Score Histogram
    plt.figure()
    sns.histplot(df['quality_score'], bins=30, kde=True)
    plt.title("Quality Score Distribution of ICBHI Training Samples (Alpha 0.7, Beta 0.3)")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig('./icbhi_training_quality_score_alpha0.7_beta0.3.png')
    plt.show()
    
    # Quality_Score Histogram
    plt.figure()
    sns.histplot(df['quality_score2'], bins=30, kde=True)
    plt.title("Quality Score Distribution of ICBHI Training Samples (Alpha 0.3, Beta 0.7)")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig('./icbhi_training_quality_score_alpha0.3_beta0.7.png')
    plt.show()
    
    '''
    # Jointplot: flatness vs RMS
    sns.jointplot(data=df, x='flatness', y='rms', kind='scatter')
    plt.suptitle("Flatness vs RMS", y=1.02)
    plt.savefig('./snubh_flat_vs_rms.png')
    plt.show()
    '''