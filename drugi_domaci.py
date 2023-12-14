import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.transform import iradon

# Parametri
R1 = 100 # cm
R2 = 25  # cm
R3 = 12  # cm
mu1 = 1  # cm^-1
mu2 = 2  # cm^-1
mu3 = 4  # cm^-1
s = 0.1  # cm
detector_size = 256


N = 180 # Broj projekcija



def napravi_fantom():
    # Parametri
    R1 = 100  # cm
    R2 = 25  # cm
    R3 = 12  # cm  
    # Kreirajte matricu
    fantom = np.zeros((256, 256))
    
    # Postavite vrednosti za prvi krug u centru
    center1 = (256 // 2, 256 // 2)
    for (i, j), value in np.ndenumerate(fantom):
        distance_squared = (i - center1[0])**2 + (j - center1[1])**2
        if distance_squared <= R1**2:
            fantom[i, j] = 0.1
    
    # Postavite vrednosti za drugi krug pomereno na levo
    center2 = (256 // 3, 256 // 3)
    for (i, j), value in np.ndenumerate(fantom):
        distance_squared = (i - center2[0])**2 + (j - center2[1])**2
        if distance_squared <= R2**2:
            fantom[i, j] += 0.2
    
    # Postavite vrednosti za treći krug pomereno na levo
    center3 = ((256 // 3) * 2, 256 // 3 * 2)
    for (i, j), value in np.ndenumerate(fantom):
        distance_squared = (i - center3[0])**2 + (j - center3[1])**2
        if distance_squared <= R3**2:
            fantom[i, j] += 0.4
    plt.imshow(fantom, cmap='gray', origin='lower', vmin=0, vmax=1)
    plt.title('Matrica sa tri kruga')
    plt.show()
    return fantom
    
def napravi_sinogram(fantom, N) :  
    theta = np.linspace(180., 0., N, endpoint=True)
    sinogram = np.zeros((max(fantom.shape), len(theta)))

    for i, angle in enumerate(theta):
        rotated_image = rotate(fantom, angle, mode='constant', cval=0, order=2, reshape=False)
        #sinogram[ : , i // N : (i+1) // N] =  np.sum(rotated_image, axis=0)
        sinogram[:,i] = np.sum(rotated_image, axis=0)
    
    return sinogram

def prikazi_sinogram(slika, title = '', xlabel = '', ylabel = ''):
    plt.imshow(np.abs(slika), cmap='gray', extent=(0, 180, 0, max(fantom.shape)), aspect='auto')
    plt.ylabel('Ugao (stepeni)')
    plt.xlabel('Projekcija')
    plt.title('Sinogram')
    plt.show()

def rekonstrukcija_slike(sinogram):
    theta = np.linspace(180., 0., sinogram.shape[1], endpoint=True)
    reconstructed_backprojection = np.zeros((sinogram.shape[0], sinogram.shape[0]))
    for i, angle in enumerate(theta):
        kao_jedan_niz = np.zeros_like(reconstructed_backprojection)
        kao_jedan_niz[: , :] = sinogram[:, i]
        rotated_projection = rotate(kao_jedan_niz, -angle, reshape=False)
        reconstructed_backprojection += rotated_projection
    # Prikaz
    plt.imshow(reconstructed_backprojection, cmap='gray', origin='lower', vmin=np.min(reconstructed_backprojection), vmax = np.max(reconstructed_backprojection))
    plt.title('Rekonstruisana slika pomoću projektovanja unazad')
    plt.show()
    
    
def rekonstrukcija_iradon(sinogram):
    theta = np.linspace(180., 0., sinogram.shape[1], endpoint=True)
    reconstructed_iradon = iradon(sinogram, theta=theta, circle=True)
    reconstructed_iradon = np.flipud(reconstructed_iradon)
    
    # Prikaz rekonstruisane slike
    plt.imshow(reconstructed_iradon, cmap='gray', origin='lower')
    plt.title('Rekonstruisana slika pomoću iradon')
    plt.show()
    return reconstructed_iradon
def filtriraj(sinogram, filter_function):
  
    num_projections, num_points = sinogram.shape
    filtered_sinogram = np.zeros_like(sinogram, dtype=np.complex128)

    # Apply filter to each projection
    for i in range(num_projections):
        filtered_sinogram[i, :] = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(sinogram[i, :])) * filter_function)

    return filtered_sinogram


def ram_lak_filter(n):
    frequencies = np.fft.fftfreq(n)
    return np.abs(frequencies)

def shepp_logan_filter(n):
    frequencies = np.fft.fftfreq(n)
    filter_values = np.sinc(frequencies) * np.hanning(n)
    return filter_values

def hann_filter(n):
    return np.hanning(n)

def butterworth_filter(n, order=5, cutoff=0.5):
    frequencies = np.fft.fftfreq(n)
    filter_values = 1 / np.sqrt(1 + (frequencies / cutoff)**(2 * order))
    return filter_values

filters = {
    'Ram-Lak': ram_lak_filter(N),
    'Shepp-Logan': shepp_logan_filter(N),
    'Hann': hann_filter(N),
    'Butterworth': butterworth_filter(N, order=5, cutoff=0.5),
}





# Glavni deo koda

fantom = napravi_fantom()
sinogram = napravi_sinogram(fantom, N)
prikazi_sinogram(sinogram)
rekonstrukcija_slike(sinogram)

filtrirani_sinogram = filtriraj(sinogram, filters['Ram-Lak'])
prikazi_sinogram(filtrirani_sinogram)
rekonstrukcija_slike(filtrirani_sinogram)

reconstructed_iradon = rekonstrukcija_iradon(sinogram)

        
