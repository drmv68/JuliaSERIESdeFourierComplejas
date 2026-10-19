# ============================================================
# TAREA 3 — Series de Fourier complejas
# Reconstrucción del contorno de un OSO con 100 coeficientes
# (n = -50, ..., 49)
#
# Paso 1: señal compleja            z(t) = x(t) + i y(t)
# Paso 2: selección de frecuencias  n ∈ {-50,...,49} (100 términos)
# Paso 3: discretización            Δt = 1/N
# Paso 4: integral -> suma de Riemann
#          c_n ≈ (1/N) Σ_{k=0}^{N-1} z_k e^(-i2πnk/N)
#          (equivale exactamente a dividir la FFT por N y tomar
#           los bines correctos — mismo resultado, O(N log N))
# Paso 5: reconstrucción
#          z_100(t) = Σ_{n=-50}^{49} c_n e^(i2πnt)
#
# Entrada: oso_contorno.csv (columnas: k, x, y), contorno externo
# de la silueta del oso, re-muestreado por longitud de arco.
# ============================================================

using DelimitedFiles, FFTW, LinearAlgebra, Plots

cd(@__DIR__)   # trabajar siempre en la carpeta donde está este script

# ---------- Paso 0: cargar el contorno ----------
datos = readdlm("oso_contorno.csv", ',', skipstart=1)
x = Float64.(datos[:, 2])
y = Float64.(datos[:, 3])

# ---------- Paso 1: señal compleja ----------
z = x .+ im .* y
N = length(z)
println("Puntos del contorno: N = $N")

# ---------- Paso 2-4: coeficientes de Fourier (vía FFT) ----------
n = -50:49                      # 100 frecuencias simétricas
X = fft(z) ./ N                 # DFT completa normalizada (todas las frecuencias 0..N-1)
c_n = X[mod.(n, N) .+ 1]        # nos quedamos solo con las 100 que nos interesan

# ---------- Paso 5: reconstrucción continua ----------
Nt = 1000
t = range(0, 1, length = Nt)
E = exp.(2im * pi .* (t .* n'))      # matriz (Nt x 100) de exponenciales
z_recon = E * c_n

# ---------- Error de reconstrucción (evaluando en los mismos puntos muestreados) ----------
t_k = (0:N-1) ./ N
E_k = exp.(2im * pi .* (t_k .* n'))
error_rel = norm(E_k * c_n .- z) / norm(z)
println("Error relativo con 100 términos: ", round(error_rel * 100, digits = 3), " %")

# ---------- Guardar los coeficientes (tabla para el informe) ----------
tabla = ["n" "Re(c_n)" "Im(c_n)" "|c_n|" "fase(rad)";
         collect(n) real(c_n) imag(c_n) abs.(c_n) angle.(c_n)]
writedlm("coeficientes_fourier_oso.csv", tabla, ',')
println("Coeficientes guardados en coeficientes_fourier_oso.csv")

# ---------- Gráfica principal: original vs. reconstrucción ----------
grafica = plot(x, y, label = "Contorno original", ls = :dash, color = :gray, lw = 1.3,
               aspect_ratio = :equal, xlabel = "Re(z)", ylabel = "Im(z)",
               title = "Fourier — Silueta de Oso (100 términos)",
               legend = :outertop, size = (750, 600))
plot!(grafica, real(z_recon), imag(z_recon),
      label = "Reconstrucción (100 términos)", lw = 2, color = :dodgerblue)
savefig(grafica, "reconstruccion_oso.png")
display(grafica)
println("Gráfica guardada en reconstruccion_oso.png")

# ---------- Comparación con distinto número de términos (discusión del informe) ----------
conteos = [10, 30, 100, 400]
subgraficas = Plots.Plot[]
for Mt in conteos
    mitad = div(Mt, 2)
    nn = -mitad:(mitad - 1)
    cc = X[mod.(nn, N) .+ 1]
    Ecc = exp.(2im * pi .* (t .* nn'))
    zz = Ecc * cc
    p = plot(x, y, color = :gray, ls = :dash, lw = 1, label = "", aspect_ratio = :equal,
              title = "$(Mt) términos", framestyle = :none)
    plot!(p, real(zz), imag(zz), color = :dodgerblue, lw = 1.6, label = "")
    push!(subgraficas, p)
end
comparacion = plot(subgraficas..., layout = (1, length(conteos)), size = (1300, 350))
savefig(comparacion, "comparacion_terminos_oso.png")
println("Comparación guardada en comparacion_terminos_oso.png")

println("--- Simulación finalizada exitosamente ---")