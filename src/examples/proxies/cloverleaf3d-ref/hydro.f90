!Crown Copyright 2012 AWE.
!
! This file is part of CloverLeaf.
!
! CloverLeaf is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the
! Free Software Foundation, either version 3 of the License, or (at your option)
! any later version.
!
! CloverLeaf is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! CloverLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Controls the main hydro cycle.
!>  @author Wayne Gaudin
!>  @details Controls the top level cycle, invoking all the drivers and checks
!>  for outputs and completion.

SUBROUTINE hydro

  USE clover_module
  USE timestep_module
  USE viscosity_module
  USE PdV_module
  USE accelerate_module
  USE flux_calc_module
  USE advection_module
  USE reset_field_module
  USE iso_c_binding
  USE conduit
  USE ascent

  use, intrinsic :: iso_c_binding, only: c_long

  IMPLICIT NONE

  interface
    ! time_t time(time_t *tloc)
    function c_time(tloc) bind(c, name='time')
        import :: c_long
        integer(kind=c_long), intent(in), value :: tloc
        integer(kind=c_long)                    :: c_time
    end function c_time
  end interface

  INTEGER         :: loc(1),err,rank,size,color,rank_split,sim_vis_comm
  INTEGER(kind=8) :: unix
  REAL(KIND=8)    :: timer,timerstart,wall_clock,step_clock,sim_timer,cycle_time

  REAL(KIND=8)    :: grind_time,cells,rstep
  REAL(KIND=8)    :: step_time,step_grind,vis_time
  REAL(KIND=8)    :: first_step,second_step
  REAL(KIND=8)    :: kernel_total,totals(parallel%max_task)

  INTEGER(kind=8), parameter :: initial_steps = 5
  REAL(kind=8), parameter :: max_sim_time = 10.0

  TYPE(C_PTR) my_ascent
  TYPE(C_PTR) ascent_opts

  my_ascent   = ascent_create()
  ascent_opts = conduit_node_create()

  CALL conduit_node_set_path_int32(ascent_opts,"mpi_comm",MPI_COMM_WORLD)
  ! To Enable Web Streaming Add, change to "true"
  CALL conduit_node_set_path_char8_str(ascent_opts,"web/stream", "false")
  CALL conduit_node_set_path_char8_str(ascent_opts,"pipeline/type", "vtkm")
  ! Runtime defined here
  CALL conduit_node_set_path_char8_str(ascent_opts,"runtime/type", "probing")
  CALL conduit_node_set_path_char8_str(ascent_opts,"actions_file", "probing_actions.yaml")
  ! other options
  CALL conduit_node_set_path_char8_str(ascent_opts,"field_filtering", "true")
  CALL ascent_open(my_ascent,ascent_opts)

  unix = c_time(int(0, kind=8))
  WRITE(g_out_stamps,*) 'start sim ', unix

  IF(MPI_COMM_NULL.NE.parallel%sim_comm)THEN
    CALL clover_barrier_sim
  ENDIF
  timerstart = timer()
  sim_timer = timerstart

  DO
    
    ! CALL ascent_timer_start(C_CHAR_"CLOVER_MAIN_LOOP"//C_NULL_CHAR)
    step_time = timer()
    
    step = step + 1

    ! only on sim nodes
    ! IF(parallel%task.LT.parallel%max_task)THEN
    IF(MPI_COMM_NULL.NE.parallel%sim_comm)THEN
      CALL timestep()
      CALL PdV(.TRUE.)
      CALL accelerate()
      CALL PdV(.FALSE.)    
      CALL flux_calc()
      CALL advection()
      CALL reset_field()
    ELSE
      CALL visit(my_ascent, 0)  ! TODO: avoid last call time+g_small.GT.end_time.OR.step.GE.end_step
      CYCLE
    ENDIF
    
    advect_x = .NOT. advect_x

    time = time + dt    ! TODO: 

    wall_clock=timer() - timerstart

    ! write node sim time to file
    WRITE(g_out_times,*) 'step time ', step, timer() - step_time

    ! CALL ascent_timer_stop(C_CHAR_"CLOVER_MAIN_LOOP"//C_NULL_CHAR)

    IF(summary_frequency.NE.0) THEN
      IF(MOD(step, summary_frequency).EQ.0)THEN
        IF(MPI_COMM_NULL.NE.parallel%sim_comm)THEN
          CALL field_summary()
        ENDIF
      ENDIF
    ENDIF

    
    ! visualization
    IF (visit_sim_time.GT.0.0 .AND. MPI_COMM_NULL.NE.parallel%sim_comm) THEN  ! WALL CLOCK TIME BASED VIS INVOCATION
      ! IF(MPI_COMM_NULL.NE.parallel%sim_comm)THEN
      CALL clover_barrier_sim
      ! ENDIF

      ! trigger vis based on cycle time
      cycle_time = timer() - sim_timer
      IF(cycle_time.GT.visit_sim_time .OR. step.EQ.1) THEN
        WRITE(g_out,*) 'step ', step
        vis_time=timer()

        IF (step.EQ.1) THEN
          cycle_time = visit_sim_time
        ENDIF

        unix = c_time(int(0, kind=8))
        WRITE(g_out_stamps,*) 'end sim ', unix
        WRITE(g_out_times,*) '       sim ', step, timer()-sim_timer

        CALL visit(my_ascent, cycle_time)

        wall_clock=timer() - timerstart
        WRITE(g_out_times,*) '       vis ', step, timer()-vis_time
        WRITE(g_out_stamps,*) 'start sim ', unix
        sim_timer = timer()
      ENDIF
    ELSE IF (visit_frequency.NE.0 ) THEN   ! STEP BASED VIS INVOCATION
      IF(MOD(step, visit_frequency).EQ.initial_steps) THEN
        vis_time=timer()

        unix = c_time(int(0, kind=8))
        WRITE(g_out_stamps,*) 'end sim ', unix
        WRITE(g_out_times,*) '       sim ', step, timer()-sim_timer

        cycle_time = timer() - sim_timer
        IF(step.LE.visit_frequency) THEN  ! extrapolate time of initial cycle
          cycle_time = cycle_time / initial_steps * visit_frequency
        ENDIF

        CALL visit(my_ascent, cycle_time)

        wall_clock=timer() - timerstart
        WRITE(g_out_times,*) '       vis ', step, timer()-vis_time
        WRITE(g_out_stamps,*) 'start sim ', unix
        sim_timer = timer()
      ENDIF
    ENDIF

    ! Sometimes there can be a significant start up cost that appears in the first step.
    ! Sometimes it is due to the number of MPI tasks, or OpenCL kernel compilation.
    ! On the short test runs, this can skew the results, so should be taken into account
    !  in recorded run times.
    IF(step.EQ.1) first_step=(timer() - step_time)
    IF(step.EQ.2) second_step=(timer() - step_time)

    ! finalizing actions after sim completion
    IF(time+g_small.GT.end_time.OR.step.GE.end_step) THEN

      unix = c_time(int(0, kind=8))
      WRITE(g_out_stamps,*) 'end sim ', unix

      complete=.TRUE.
      IF(MPI_COMM_NULL.NE.parallel%sim_comm)THEN
        CALL field_summary()
      ENDIF
    ! IF(visit_frequency.NE.0) CALL visit()

      wall_clock=timer() - timerstart
      IF ( parallel%boss ) THEN
        WRITE(g_out,*)
        WRITE(g_out,*) 'Calculation complete'
        WRITE(g_out,*) 'Clover is finishing'
        WRITE(g_out,*) 'Wall clock ', wall_clock
        WRITE(g_out,*) 'First step overhead', first_step-second_step
        WRITE(    0,*) 'Wall clock ', wall_clock
        WRITE(    0,*) 'First step overhead', first_step-second_step
      ENDIF

      IF ( .FALSE. ) THEN
        ! First we need to find the maximum kernel time for each task. This
        ! seems to work better than finding the maximum time for each kernel and
        ! adding it up, which always gives over 100%. I think this is because it
        ! does not take into account compute overlaps before syncronisations
        ! caused by halo exhanges.
        kernel_total=profiler%timestep+profiler%ideal_gas+profiler%viscosity+profiler%PdV          &
                    +profiler%revert+profiler%acceleration+profiler%flux+profiler%cell_advection   &
                    +profiler%mom_advection+profiler%reset+profiler%halo_exchange+profiler%summary &
                    +profiler%visit
        CALL clover_allgather(kernel_total,totals)
        ! So then what I do is use the individual kernel times for the
        ! maximum kernel time task for the profile print
        loc=MAXLOC(totals)
        kernel_total=totals(loc(1))
        CALL clover_allgather(profiler%timestep,totals)
        profiler%timestep=totals(loc(1))
        CALL clover_allgather(profiler%ideal_gas,totals)
        profiler%ideal_gas=totals(loc(1))
        CALL clover_allgather(profiler%viscosity,totals)
        profiler%viscosity=totals(loc(1))
        CALL clover_allgather(profiler%PdV,totals)
        profiler%PdV=totals(loc(1))
        CALL clover_allgather(profiler%revert,totals)
        profiler%revert=totals(loc(1))
        CALL clover_allgather(profiler%acceleration,totals)
        profiler%acceleration=totals(loc(1))
        CALL clover_allgather(profiler%flux,totals)
        profiler%flux=totals(loc(1))
        CALL clover_allgather(profiler%cell_advection,totals)
        profiler%cell_advection=totals(loc(1))
        CALL clover_allgather(profiler%mom_advection,totals)
        profiler%mom_advection=totals(loc(1))
        CALL clover_allgather(profiler%reset,totals)
        profiler%reset=totals(loc(1))
        CALL clover_allgather(profiler%halo_exchange,totals)
        profiler%halo_exchange=totals(loc(1))
        CALL clover_allgather(profiler%summary,totals)
        profiler%summary=totals(loc(1))
        CALL clover_allgather(profiler%visit,totals)
        profiler%visit=totals(loc(1))

        IF ( parallel%boss ) THEN
          WRITE(g_out,*)
          WRITE(g_out,'(a58,2f16.4)')"Profiler Output                 Time            Percentage"
          WRITE(g_out,'(a23,2f16.4)')"Timestep              :",profiler%timestep,100.0*(profiler%timestep/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Ideal Gas             :",profiler%ideal_gas,100.0*(profiler%ideal_gas/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Viscosity             :",profiler%viscosity,100.0*(profiler%viscosity/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"PdV                   :",profiler%PdV,100.0*(profiler%PdV/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Revert                :",profiler%revert,100.0*(profiler%revert/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Acceleration          :",profiler%acceleration,100.0*(profiler%acceleration/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Fluxes                :",profiler%flux,100.0*(profiler%flux/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Cell Advection        :",profiler%cell_advection,100.0*(profiler%cell_advection/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Momentum Advection    :",profiler%mom_advection,100.0*(profiler%mom_advection/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Reset                 :",profiler%reset,100.0*(profiler%reset/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Halo Exchange         :",profiler%halo_exchange,100.0*(profiler%halo_exchange/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Summary               :",profiler%summary,100.0*(profiler%summary/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Visit                 :",profiler%visit,100.0*(profiler%visit/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"Total                 :",kernel_total,100.0*(kernel_total/wall_clock)
          WRITE(g_out,'(a23,2f16.4)')"The Rest              :",wall_clock-kernel_total,100.0*(wall_clock-kernel_total)/wall_clock
        ENDIF
      ENDIF ! IF(FALSE)

      CALL ascent_close(my_ascent)
      CALL ascent_destroy(my_ascent)
      CALL clover_finalize

      EXIT

    END IF ! actions after sim completion

    ! print out stats
    IF (parallel%boss) THEN
      wall_clock=timer()-timerstart
      step_clock=timer()-step_time
      WRITE(g_out,*)"Wall clock ",wall_clock
      WRITE(0    ,*)"Wall clock ",wall_clock
      cells = grid%x_cells * grid%y_cells
      rstep = step
      grind_time   = wall_clock/(rstep * cells)
      step_grind   = step_clock/cells
      WRITE(0    ,*)"Average time per cell ",grind_time
      WRITE(g_out,*)"Average time per cell ",grind_time
      WRITE(0    ,*)"Step time per cell    ",step_grind
      WRITE(g_out,*)"Step time per cell    ",step_grind
    END IF  

  END DO

END SUBROUTINE hydro
