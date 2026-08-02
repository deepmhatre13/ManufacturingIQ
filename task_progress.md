# ManufacturingIQ Production Readiness - Task Progress

## Phase 1: Repository Audit & Dependency Analysis
- [x] Read all source files and understand import graph
- [x] Analyze requirements.txt, requirements-dev.txt, requirements-lock.txt
- [x] Map all imports to packages
- [x] Identify dependency conflicts

## Phase 2: Fix requirements.txt (Runtime Dependencies)
- [ ] Create clean requirements.txt with only runtime packages
- [ ] Add langgraph ecosystem with compatible versions
- [ ] Remove training-only packages (evidently, scipy → dev)
- [ ] Remove unused packages (markdown, weasyprint)
- [ ] Keep shap (used at runtime in explanation_agent)
- [ ] Keep fpdf2 (used at runtime in reports/generator.py)

## Phase 3: Fix requirements-dev.txt
- [ ] Move training/monitoring packages here
- [ ] Keep testing tools
- [ ] Keep dev tools (black, ruff, jupyter)

## Phase 4: Fix Dockerfile
- [ ] Multi-stage build
- [ ] Optimize layer caching
- [ ] Non-root user
- [ ] Healthcheck
- [ ] Reduce image size

## Phase 5: Fix render.yaml
- [ ] Verify build/start commands
- [ ] Verify environment variables

## Phase 6: Code Fixes
- [ ] Fix any import issues
- [ ] Fix configuration issues
- [ ] Fix security issues (secrets)

## Phase 7: Verification
- [ ] pip install -r requirements.txt
- [ ] pip check
- [ ] pytest
- [ ] Docker build
- [ ] Documentation update