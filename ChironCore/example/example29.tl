:base = 150
:angle = 60
:depth = 4
:step = 0
:dir = 1
:growth = 2
:decay = 1
:limit = 7
:alternate = 0
:spiralbase = 10
:geofactor = 3/2
:offset = 20
:shadow = 0

pendown
repeat :depth [
  if (:alternate < 2) [
    repeat 6 [
      forward :base
      right :angle
      :step = :step + 1
      
      if (:step > 3) [
        penup
        goto (:base * :geofactor, :base / :geofactor)
        pendown
        :geofactor = :geofactor + (1/4)
        
        repeat 3 [
          left 120
          forward (:base - :offset)
          :offset = :offset + 5
          if (:offset > 40) [
            :offset = 10
            right 60
            forward (:base / 2)
          ]
        ]
      ] else [
        backward (:base / 3)
        left 45
        :shadow = :shadow + :base
      ]
    ]
    
    if (:shadow > 100) [
      :base = :base + (:growth * 2)
      :growth = :growth + :decay
    ] else [
      :base = :base - (:growth * 3)
      :decay = :decay + 1
    ]
  ] else [
    :spiralsteps = :spiralbase
    :spiralangle = 5
    repeat 8 [
      forward :spiralsteps
      right :spiralangle
      :spiralsteps = :spiralsteps * (4/3)
      :spiralangle = :spiralangle + 3
      
      if (:spiralsteps > 50) [
        penup
        goto (:spiralsteps, :spiralangle)
        pendown
        repeat 2 [
          left 90
          forward (:spiralsteps / 2)
          :spiralbase = :spiralbase + 5
        ]
      ]
    ]
    :alternate = -1
  ]

  :temp = :base
  :base = :base + (:step * 2)
  :step = :temp - (:offset / 2)
  :limit = :limit + (:growth * :dir)
  
  if (:base > 250) [
    :base = 50
    :dir = :dir * -1
    repeat 2 [
      right 180
      forward 30
      :geofactor = :geofactor - (1/2)
    ]
  ] else [
    left 45
    forward (:base * (3/4))
  ]

  penup
  goto (0, 0)
  pendown
  :alternate = :alternate + 1
  :depth = :depth - 1
]
penup