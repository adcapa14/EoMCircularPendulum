import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del sistema
g = 9.81      # Gravedad (m/s^2)
m = 1.0       # Masa del péndulo (kg)
L = 1.25       # Longitud del péndulo (m)
r = 0.5       # Radio de la circunferencia (m)
omega = 2.0   # Velocidad angular del soporte (rad/s)

# Ecuación diferencial
def pendulum_motion(t, y):
    theta, theta_dot = y  # y[0] = theta, y[1] = theta_dot
    phi = omega * t
    theta_ddot = - (g / L) * np.sin(theta) - (r * omega**2 / L) * np.cos(theta) * np.cos(phi - theta)
    return [theta_dot, theta_ddot]

# Condiciones iniciales
theta0 = 0.1  # Ángulo inicial (rad)
theta_dot0 = 0.0  # Velocidad angular inicial (rad/s)
y0 = [theta0, theta_dot0]

# Tiempo de simulación
t_span = (0, 10)  # Desde 0 a 10 segundos
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Puntos de evaluación

# Resolver la ecuación diferencial
sol = solve_ivp(pendulum_motion, t_span, y0, t_eval=t_eval, method='RK45')

# Extraer soluciones
t = sol.t
theta = sol.y[0]
theta_dot = sol.y[1]

# Calcular energías
phi = omega * t
x_a = r * np.cos(phi)
y_a = r * np.sin(phi)

x_p = x_a + L * np.sin(theta)
y_p = y_a - L * np.cos(theta)

kinetic_energy = 0.5 * m * (L * theta_dot)**2
potential_energy = m * g * (y_p - y_p.min())  # Ajustar a energía potencial mínima
total_energy = kinetic_energy + potential_energy

# Graficar resultados
plt.figure(figsize=(12, 8))

# Ángulo vs tiempo
plt.subplot(2, 2, 1)
plt.plot(t, theta, label=r"$\theta(t)$")
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo (rad)")
plt.title("Evolución del ángulo")
plt.legend()
plt.grid()

# Velocidad angular vs tiempo
plt.subplot(2, 2, 2)
plt.plot(t, theta_dot, label=r"$\dot{\theta}(t)$", color='orange')
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad angular (rad/s)")
plt.title("Velocidad angular")
plt.legend()
plt.grid()

# Energías
plt.subplot(2, 2, 3)
plt.plot(t, kinetic_energy, label="Energía cinética", color='green')
plt.plot(t, potential_energy, label="Energía potencial", color='red')
plt.plot(t, total_energy, label="Energía total", color='blue', linestyle='--')
plt.xlabel("Tiempo (s)")
plt.ylabel("Energía (J)")
plt.title("Energías del sistema")
plt.legend()
plt.grid()

# Trayectoria del péndulo
plt.subplot(2, 2, 4)
plt.plot(x_a, y_a, label="Soporte", color='purple')
plt.plot(x_p, y_p, label="Péndulo", color='brown')
plt.xlabel("Posición x (m)")
plt.ylabel("Posición y (m)")
plt.title("Trayectoria del péndulo")
plt.legend()
plt.axis('equal')
plt.grid()

plt.tight_layout()
plt.show()