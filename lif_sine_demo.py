# Demo snnTorch: LIF con raster plot (entrada temporal)
import torch
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt

# 1) Señal senoidal (temporal)
T = 200
x = torch.linspace(0, 4*torch.pi, T)
signal = 0.5*(torch.sin(x) + 1.0)              # en [0,1], tamaño [T]

# 2) Codificación a spikes (RATE) indicando que es entrada temporal
#    Shape esperado: [batch, features, time]
spike_in = spikegen.rate(signal.unsqueeze(0).unsqueeze(0), time_var_input=True)

# 3) Neurona LIF
lif = snn.Leaky(beta=0.9, threshold=0.5)

# 4) Simulación temporal
mem = torch.zeros(1, 1)
spikes_out = []
for t in range(T):
    spk, mem = lif(spike_in[:, :, t], mem)
    spikes_out.append(spk)

spikes_out = torch.stack(spikes_out, dim=2).squeeze()   # [T]
print(f"Tasa de disparo: {spikes_out.float().mean().item() * 100:.1f}%")

# 5) Raster plot
plt.figure(figsize=(7, 2))
t_idx = torch.where(spikes_out > 0)[0].cpu()
plt.eventplot(t_idx.tolist(), lineoffsets=1, linelengths=0.9)
plt.yticks([])
plt.xlabel("Tiempo (pasos)")
plt.title("Raster de spikes (salida LIF)")
plt.tight_layout()
plt.show()
