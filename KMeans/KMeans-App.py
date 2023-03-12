import pygame
from random import randint
import math
from sklearn.cluster import KMeans


def distance(p1, p2):
	return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0])  +  (p1[1] - p2[1]) * (p1[1] - p2[1]))

pygame.init()

screen = pygame.display.set_mode((1200,700))

pygame.display.set_caption("KMeans of Ngọc Sơn")
running = True
clock = pygame.time.Clock()


BACKGROUND = (220, 220, 220)
BLACK = (0, 0, 0)
BACKGROUND_PANEL = (249, 255, 230)
WHITE = (255, 255, 255)

# Create Color

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (147, 153, 35)
PURPLE = (255, 0, 255)
SKY = (0, 255, 255)
ORANGE = (255, 125, 25)
GRAPE = (100, 25, 125)
GRASS = (55, 155, 65)
PINK = (255, 179, 218)

COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE, SKY, ORANGE, GRAPE, GRASS, PINK]


# Create Font
# Tạo Font và Chữ
font_big = pygame.font.SysFont('sans', 60)
font = pygame.font.SysFont('sans', 40)
font_small = pygame.font.SysFont('sans', 20)


# Tạo kí tự
text_minus = font.render('-', True, BLACK)
text_plus = font.render('+', True, BLACK)
text_run = font.render('Run', True, BLACK)
text_random = font.render('Random', True, BLACK)
text_algori = font.render('Algorithm', True, BLACK)
text_reset = font.render('Reset', True, BLACK)


# Khai báo
K = 0
error = 0
points = []
clusters = []
labels = []


while running:
	clock.tick(90)
	screen.fill(BACKGROUND)
	Mouse_x, Mouse_y = pygame.mouse.get_pos()

	# Draw interface
	# Draw title
	pygame.draw.rect(screen, BLACK, (20, 20, 1160, 80))
	pygame.draw.rect(screen, BACKGROUND_PANEL, (25, 25, 1150, 70))
	text_title = font_big.render("Kmeans Clustering", True, BLACK)
	screen.blit(text_title, (220,20))
	text_author = font.render("Ngoc Son", True, BLACK)
	screen.blit(text_author, (930,35))

	# Draw panel
	pygame.draw.rect(screen, BLACK, (20, 120, 800, 500))
	pygame.draw.rect(screen, BACKGROUND_PANEL,(25, 125, 790, 490))
	
	# Tính năng
	pygame.draw.rect(screen, BLACK, (850, 120, 330, 500))
	pygame.draw.rect(screen, BACKGROUND_PANEL,(855, 125, 320, 490))

	
	# Draw Button
	
	# Buton -
	pygame.draw.rect(screen, BLACK, (885, 145, 50, 50))
	pygame.draw.rect(screen, BACKGROUND,(887, 147, 46, 46))
	screen.blit(text_minus,(905, 143))
	
	# Buton + 
	pygame.draw.rect(screen, BLACK, (1005, 145, 50, 50))
	pygame.draw.rect(screen, BACKGROUND,(1007, 147, 46, 46))
	screen.blit(text_plus,(1020,145))
	
	# K
	text_k = font.render("K = " + str(K), True, BLACK)
	screen.blit(text_k, (1075, 145))

	# Button Run
	pygame.draw.rect(screen, BLACK, (885, 225, 170, 50))
	pygame.draw.rect(screen, BACKGROUND,(887, 227, 166, 46))
	screen.blit(text_run,(935,225))

	# Button Random
	pygame.draw.rect(screen, BLACK, (885, 305, 170, 50))
	pygame.draw.rect(screen, BACKGROUND,(887, 307, 166, 46))
	screen.blit(text_random,(910,305))

	# Button Algorithm
	pygame.draw.rect(screen, BLACK, (885, 465, 170, 50))
	pygame.draw.rect(screen, BACKGROUND,(887, 467, 166, 46))
	screen.blit(text_algori,(900,465))
	
	# Button Reset
	pygame.draw.rect(screen, BLACK, (885, 545, 170, 50))
	pygame.draw.rect(screen, BACKGROUND,(887, 547, 166, 46))
	screen.blit(text_reset,(925,545))

	
	# Draw mouse position when mouse is in panel
	if 30 < Mouse_x < 810 and 130 < Mouse_y < 610:
		text_mouse = font_small.render("(" + str(Mouse_x - 31) + " , " + str(Mouse_y - 131) + ")", True, BLACK)
		screen.blit(text_mouse, (Mouse_x + 10, Mouse_y + 10))

	# Draw cluster
	# if 35 < Mouse_x < 805 and 135 < Mouse_y < 605:

	# End draw interface


	# Button
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.MOUSEBUTTONDOWN:

			# Listen click on panel/ Help create point on panel
			if 30 < Mouse_x < 810 and 130 < Mouse_y < 610:
				labels = []
				point = [Mouse_x, Mouse_y]
				points.append(point)


			if 887 < Mouse_x < 933 and 150 < Mouse_y < 190:
				if K > 0:
					K -= 1

			if 1007 < Mouse_x < 1053 and 150 < Mouse_y < 190:
				if K < 10:
					K += 1

			# Run Button
			if 887 < Mouse_x  < 1053 and 227 < Mouse_y < 273:
				# reset labels
				labels = []

				if clusters == []:
					print("Bạn cần random cluster")
					continue

				# lấy ra vị trí của từng điểm trong point
				# for i in range(len(points)): 
				# lấy ra từng điểm trong points
				for p in points:
					distance_to_clusters = []
					for c in clusters:
						distance_to_cluster = distance(p, c)
						distance_to_clusters.append(distance_to_cluster)
					
					min_distance = min(distance_to_clusters)
					label = distance_to_clusters.index(min_distance) 
					labels.append(label)

				# Update clusters
				# Chuyển điểm vào giữa
				for i in range(K):
					sum_x = 0
					sum_y = 0
					count = 0
					for j in range(len(points)):
						if labels[j] == i:
							sum_x += points[j][0]
							sum_y += points[j][1]
							count += 1
					if count != 0:
						new_cluster_x = sum_x/count
						new_cluster_y = sum_y/count
						clusters[i] = [new_cluster_x, new_cluster_y]

			# Random Button
			if 887 < Mouse_x  < 1053 and 307 < Mouse_y < 353:
				labels = []
				clusters = []
				for i in range(K):
					random_point = [randint(35, 805), randint(135, 605)]
					clusters.append(random_point)

			# Button Algorithm
			if 887 < Mouse_x  < 1053 and 467 < Mouse_y < 513:
				try:
					kmeans = KMeans(n_clusters=K).fit(points)
					labels = kmeans.predict(points)
					clusters = kmeans.cluster_centers_
				except:
					print("Error")

			# Button Reset
			if 887 < Mouse_x  < 1053 and 547 < Mouse_y < 593:
				points = []
				clusters = []
				labels = []
				K = 0
				error = 0

	# Draw point
	for i in range(len(points)):
		pygame.draw.circle(screen, BLACK, (points[i][0], points[i][1]), 5)
		if labels == [] :
			pygame.draw.circle(screen, WHITE, (points[i][0], points[i][1]), 4)
		else:
			pygame.draw.circle(screen, COLORS[labels[i]], (points[i][0], points[i][1]), 4)
			
	# Draw cluster
	for i in range(len(clusters)):
		pygame.draw.circle(screen, COLORS[i], (int(clusters[i][0]), int(clusters[i][1])), 10)


	# Đưa ra lỗi....
	error = 0
	if clusters != [] and labels != []:
		for i in range(len(points)):
			error += distance(points[i], clusters[labels[i]] )

	text_error = font.render("Error = " + str(int(error)), True, BLACK)
	screen.blit(text_error, (910, 385))


	pygame.display.flip()
pygame.quit()

