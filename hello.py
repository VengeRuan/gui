import random
import tkinter as tk

WIDTH, HEIGHT = 480, 640
GROUND = HEIGHT - 70
BIRD_X = 110
BIRD_SCALE = 1.25
# Collision bounds match the bird's scaled visible outline (including its beak).
BIRD_LEFT = -15 * BIRD_SCALE
BIRD_RIGHT = 25 * BIRD_SCALE
BIRD_TOP = -12 * BIRD_SCALE
BIRD_BOTTOM = 14 * BIRD_SCALE


class FlappyBird:
    def __init__(self, root):
        self.root = root
        self.root.title("Tkinter Flappy Bird")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#70c5ce", highlightthickness=0)
        self.canvas.pack()
        self.root.bind("<space>", self.flap)
        self.root.bind("<Button-1>", self.flap)
        self.reset()
        self.draw()

    def reset(self):
        self.bird_y = HEIGHT / 2
        self.velocity = 0.0
        self.pipes = []
        self.score = 0
        self.frame = 0
        self.running = False
        self.game_over = False

    def flap(self, _event=None):
        if self.game_over:
            self.reset()
        if not self.running:
            self.running = True
        self.velocity = -8.2

    def add_pipe(self):
        gap = 160
        center = random.randint(170, GROUND - 150)
        self.pipes.append([WIDTH + 25, center - gap / 2, center + gap / 2, False])

    def update(self):
        if not self.running:
            return
        self.frame += 1
        self.velocity += 0.48
        self.bird_y += self.velocity
        if self.frame % 95 == 0:
            self.add_pipe()

        for pipe in self.pipes:
            pipe[0] -= 3.2
            if not pipe[3] and pipe[0] + 62 < BIRD_X:
                pipe[3] = True
                self.score += 1
        self.pipes = [pipe for pipe in self.pipes if pipe[0] + 62 > 0]

        bird_left, bird_right = BIRD_X + BIRD_LEFT, BIRD_X + BIRD_RIGHT
        bird_top, bird_bottom = self.bird_y + BIRD_TOP, self.bird_y + BIRD_BOTTOM
        hit = bird_top <= 0 or bird_bottom >= GROUND
        for x, top, bottom, _ in self.pipes:
            if bird_right > x and bird_left < x + 62 and (bird_top < top or bird_bottom > bottom):
                hit = True
        if hit:
            self.running = False
            self.game_over = True

    def draw(self):
        self.update()
        c = self.canvas
        c.delete("all")
        for x, top, bottom, _ in self.pipes:
            c.create_rectangle(x, 0, x + 62, top, fill="#45a832", outline="#247222", width=3)
            c.create_rectangle(x - 5, top - 20, x + 67, top, fill="#59bd3d", outline="#247222", width=3)
            c.create_rectangle(x, bottom, x + 62, GROUND, fill="#45a832", outline="#247222", width=3)
            c.create_rectangle(x - 5, bottom, x + 67, bottom + 20, fill="#59bd3d", outline="#247222", width=3)
        c.create_rectangle(0, GROUND, WIDTH, HEIGHT, fill="#ded895", outline="")
        c.create_oval(BIRD_X - 15 * BIRD_SCALE, self.bird_y - 12 * BIRD_SCALE, BIRD_X + 15 * BIRD_SCALE, self.bird_y + 14 * BIRD_SCALE,
                      fill="#ffd93b", outline="#dc9922", width=2)
        c.create_oval(BIRD_X + 5 * BIRD_SCALE, self.bird_y - 7 * BIRD_SCALE, BIRD_X + 12 * BIRD_SCALE, self.bird_y, fill="white", outline="")
        c.create_oval(BIRD_X + 8 * BIRD_SCALE, self.bird_y - 5 * BIRD_SCALE, BIRD_X + 11 * BIRD_SCALE, self.bird_y - 2 * BIRD_SCALE, fill="black", outline="")
        c.create_polygon(BIRD_X + 14 * BIRD_SCALE, self.bird_y, BIRD_X + 25 * BIRD_SCALE, self.bird_y + 5 * BIRD_SCALE, BIRD_X + 14 * BIRD_SCALE, self.bird_y + 8 * BIRD_SCALE,
                         fill="#f38b2b", outline="#b85f16")
        c.create_text(WIDTH / 2, 55, text=str(self.score), font=("Arial", 32, "bold"), fill="white")
        if not self.running:
            message = "Click or press Space to fly" if not self.game_over else "Game over! Click or press Space to restart"
            c.create_text(WIDTH / 2, HEIGHT / 2 - 80, text=message, font=("Arial", 16, "bold"), fill="white")
        self.root.after(20, self.draw)


if __name__ == "__main__":
    root = tk.Tk()
    FlappyBird(root)
    root.mainloop()

