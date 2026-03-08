Yes, several options for remote amd64 builds:         
  ┌──────────────┬───────────┬───────┬─────────────────┐
  │    Option    │   Cost    │ Setup │      Notes      │
  │              │           │  Time │                 │
  ├──────────────┼───────────┼───────┼─────────────────┤
  │ GitHub       │           │ ~5    │ Best option -   │
  │ Actions      │ Free      │ min   │ native amd64    │
  │              │           │       │ runners         │
  ├──────────────┼───────────┼───────┼─────────────────┤
  │ AWS EC2      │ ~$0.10/hr │ ~10   │ t3.large or     │
  │              │           │ min   │ bigger          │
  ├──────────────┼───────────┼───────┼─────────────────┤
  │ Google Cloud │           │ ~10   │ 120             │
  │  Build       │ Free tier │ min   │ build-min/day   │
  │              │           │       │ free            │
  ├──────────────┼───────────┼───────┼─────────────────┤
  │ DigitalOcean │ $0.01/hr  │ ~5    │ Simple droplets │
  │              │           │ min   │                 │
  └──────────────┴───────────┴───────┴─────────────────┘
  GitHub Actions is the easiest - push Dockerfile to    
  repo, it builds and pushes to Docker Hub              
  automatically.                                        
                                                        
  Want me to set up a GitHub Actions workflow? It would:
  1. Trigger on push to the repo                        
  2. Build on native amd64 runner                       
  3. Push to veylansolmira/caml-env:latest  


  - alpaca = Fine-tuning on Stanford Alpaca dataset     
  (general instruction-following)                       
  - medai = Likely "medical AI" domain-specific data    
  (need to confirm with Jasmine)                        
  - sdf = Synthetic Document Finetuning (CaML's core    
  method - synthetic compassionate documents)           
  - negai/fullai = Probably different synthetic data    
  variants (negative AI? full AI scenarios?) 