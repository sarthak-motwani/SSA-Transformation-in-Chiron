:basesize = 120
:sides = 6
:angle = 60
:depth = 3
:colors = 6
:colorshift = 30
:thickness = 3
:spacing = 25
:spiralturns = 8
:spiralstep = 5
:spiralgrowth = 102/100
:flowerpoints = 12
:flowersize = 80
:starpoints = 5
:starsize = 150
:starturn = 144
:symmetry = 8
:goldenratio = 161/100


penup
goto (0, 0)
pendown

repeat :symmetry [
  :currentsize = :basesize
  :currentcolor = 1
  
  repeat :sides [
    pencolor (:currentcolor * :colorshift, 150, 255 - :currentcolor * :colorshift)
    pensize (:thickness)
    forward :currentsize
    right (360 / :sides)
    :currentcolor = :currentcolor + 1
    
    if (:currentcolor > :colors) [
      :currentcolor = 1
    ]
  ]
  
  
  penup
  goto (0, 0)
  pendown
  right (360 / :symmetry)
]


pensize (2)
:spiralsize = :basesize / 4
pencolor (180, 100, 255)

repeat :spiralturns [
  repeat 60 [
    forward :spiralsize
    right :spiralstep
    :spiralsize = :spiralsize * :spiralgrowth
  ]
  
 
  penup
  goto (0, 0)
  right (360 / :spiralturns)
  pendown
]

pensize (1)
penup
goto (0, 0)
pendown
pencolor (255, 180, 180)

repeat :flowerpoints [
  forward :flowersize
  backward :flowersize
  right (360 / :flowerpoints)
]


pensize (2)
pencolor (255, 220, 50)

repeat 2 [
  repeat :starpoints [
    forward :starsize
    right :starturn
  ]
  
  :starsize = :starsize / :goldenratio
  pencolor (255, 150, 50)
]

# Add final accents
pensize (1)
:accentsize = :basesize / 3

repeat 36 [
  pencolor (100 + :accentsize, 150, 200)
  forward :accentsize
  backward :accentsize
  right 10
  :accentsize = :accentsize + 1
  
  if (:accentsize > :basesize) [
    :accentsize = :basesize / 3
  ]
]

penup
