def visualize(trained_model, dataset, index = 0, n_neighen = 10)
    mean = dataset.mean
    std = dataset.std
    idx = index

    num_neighen = n_neighen
    neighen = []

    input_seq, target_seq = dataset[idx]
    input_seq = input_seq.unsqueeze(0).to('cpu')
    target_seq = target_seq.numpy()
    target_seq = target_seq * std + mean

    for i in range(num_neighen):
      recon_seq, _, _ = trained_model(input_seq)
      recon_seq = recon_seq.squeeze(0).detach().numpy()
      recon_seq = recon_seq * std + mean

      if recon_seq.shape != target_seq.shape:
          print(f"Shape mismatch for trajectory {idx + 1}:")
          print(f"Original shape: {target_seq.shape}")
          print(f"Reconstructed shape: {recon_seq.shape}")
      neighen.append(recon_seq)

    mse = np.mean((recon_seq - target_seq) ** 2)

    plt.figure(figsize=(8, 6))
    plt.plot(target_seq[:, 0], target_seq[:, 1], 'b-', label='Original Trajectory')

    for i in range(num_neighen):
      recon_seq_plt = neighen[i]
      plt.plot(recon_seq_plt[:, 0], recon_seq_plt[:, 1], 'r--', label='Reconstructed Trajectory' + str(i))
    plt.legend()

    plt.title(f"Trajectory {idx + 1} - MSE: {mse:.4f}")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")

    plt.show()
