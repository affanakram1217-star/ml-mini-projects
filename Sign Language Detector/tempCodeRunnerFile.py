if hands:
        for hand in hands:   #everything inside loop
            x, y, w, h = hand['bbox']

            if w == 0 or h == 0:
                continue