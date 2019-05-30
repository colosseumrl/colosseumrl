#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, overflowcheck=False, nonecheck=False, cdivision=True

cpdef void next_state_inplace(long[:, ::1] board,
                              long[::1] heads,
                              long[::1] directions,
                              long[::1] deaths,
                              const long[::1] actions):
    cdef long N = board.shape[0]
    cdef long num_players = heads.shape[0]

    cdef long x, y, i
    cdef long action, direction
    cdef long enemy_player

    for i in range(num_players):
        if deaths[i] > 0:
            continue

        # Get current head location on grid
        # We use row-wise indexing
        x = heads[i] % N
        y = heads[i] // N

        # Action is received as relative direction
        # 0 - Forward
        # 1 - Right
        # -1 - Left
        action = actions[i]

        # Convert this into cardinal directions
        direction = (directions[i] + action + 4) % 4

        # Compute the new move location for this head
        if direction == 0: # North
            y = y - 1
        elif direction == 1: # West
            x = x + 1
        elif direction == 2: # South
            y = y + 1
        elif direction == 3: # East
            x = x - 1

        # Update our player's direction
        directions[i] = direction

        # If we have crashed into the wall, then we have killed ourselves
        if (x < 0) or (x >= N) or (y < 0) or (y >= N):
            deaths[i] = i + 1

        # If we have crashed into another player, then they have killed us
        elif board[y, x] > 0:
            enemy_player = board[y, x]
            deaths[i] = enemy_player

            # If we have crashed into their head, then we both die
            if heads[enemy_player - 1] == (N * y + x):
                deaths[enemy_player - 1] = i + 1

        # Otherwise we move normally
        else:
            board[y, x] = i + 1
            heads[i] = N * y + x


cpdef void relative_player_inplace(long[:, ::1] board, const long num_players, const long player):
    cdef long N = board.shape[0]

    for i in range(N):
        for j in range(N):
            if board[i, j] > 0:
                board[i, j] = ((board[i, j] - player + num_players) % num_players) + 1








