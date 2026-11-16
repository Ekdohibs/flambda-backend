(* Proofs of the statements of FlambdaRegionsCallsSpec.v. Nothing here
   needs review.

   - The forms of a step ([region_step_forms]):
     FlambdaRegionsStepsProofs.region_step_cases with what the test of
     the simples gives at each form.
   - The frames stay in the order of their marks
     ([region_step_keeps_frames_ordered]). That part is C2's.
   - Where a call, or the application of a continuation, lands (the
     section [Call_chain]): the chain of
     FlambdaSemanticsProofs.call_function_values_all, with a property
     of the values that code is given, one of the values that a
     continuation is given, one of a continuation and one of the
     variables and the continuations of the state landed on, each
     abstract. It is used twice.
   - A mark is held by variables only ([region_step_marks]): a step of
     a state that has the invariant keeps the first part of the test,
     on its own. The heap is in that part: no value that a cell holds
     is a mark or holds one ([heap_no_mark]), and a [Let] that is run
     on the heap keeps it so ([exec_named_no_mark]).
   - The other parts ([region_step_calls]), given the first two. The
     end of a region is where the order of the frames and the age of
     what is held are used ([end_region_usable]): a location that is
     held and that the state lists has a mark smaller than that of the
     region, so the frame of its cell stays ([on_stack_below]). A
     local allocation of a cell lists a location that the heap did not
     bind, which is none of the values that are held
     ([local_cell_allocation_calls]).
   - The statements. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaHeapProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaClosuresSpec FlambdaClosuresProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsErasureSpec
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsCellsProofs
  FlambdaRegionsNoPoisonProofs
  FlambdaRegionsOnceProofs FlambdaRegionsInvariantProofs
  FlambdaRegionsErasedProofs FlambdaRegionsPlainStepProofs
  FlambdaRegionsErasureProofs FlambdaRegionsCallsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The forms of a step, with what the test of the simples gives             *)
(* ======================================================================== *)

(* FlambdaRegionsStepsProofs.region_step_cases, with three more facts:
   the end of a region is the [Let] of a primitive of regions applied
   to the simple of the region; where a region is given to a
   continuation, one of the arguments is a mark that is on the stack
   and the test of the simples passed; and at every other expression
   that [step] runs, each simple that the step evaluates may be
   evaluated. What region_step_cases says of the local values among
   the simples of a [Let] and of a [Let_static], and of the test of a
   store, is left out: nothing here needs it. *)
Section Region_step_forms.

Variable rs : region_state.
Variable Q : region_step_result -> Prop.

Hypothesis Q_undefined : ∀ fault : region_fault, Q (Region_undefined fault).
Hypothesis Q_begin :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (mark_kind : mark_kind)
    (body : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    Q (begin_region (count_uses rs) x kind mark_kind body).
Hypothesis Q_end :
  ∀ (x : variable) (kind : flambda_kind) (p : region_primitive)
    (ended_by : mark_kind) (region : simple) (body : expr),
    state_expr (rs_state rs)
    = Let x kind (Prim (Region_primitive p) [region]) body ->
    Q (end_region (count_uses rs) x kind ended_by region body).
Hypothesis Q_local :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
    (y : variable) (region : option simple) (body' : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    marker_of x body = Marker y region body' ->
    forallb (simple_usable rs) (simples_of_named d) = true ->
    is_cell_allocation d && kind_eqb kind Value = false ->
    Q (local_allocation (count_uses rs) x y kind d region body').
Hypothesis Q_local_cell :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
    (y : variable) (region : option simple) (body' : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    marker_of x body = Marker y region body' ->
    forallb (simple_usable rs) (simples_of_named d) = true ->
    is_cell_allocation d = true ->
    Q (local_cell_allocation (count_uses rs) x y d region body').
Hypothesis Q_let :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    forallb (simple_usable rs) (simples_of_named d) = true ->
    Q (plain_step (count_uses rs)).
Hypothesis Q_static :
  ∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) = Let_static bindings body ->
    forallb (simple_usable rs) (flat_map simples_of_binding bindings)
    = true ->
    Q (static_step (count_uses rs)).
Hypothesis Q_region_apply_cont :
  ∀ ac : apply_cont,
    state_expr (rs_state rs) = Apply_cont ac ->
    existsb (simple_open_mark rs) (ac_args ac) = true ->
    simples_fault rs (ac_args ac) (region_places rs) = None ->
    Q (region_apply_cont (count_uses rs) ac).
Hypothesis Q_plain :
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    state_expr (rs_state rs) <> Let x kind d body) ->
  (∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) <> Let_static bindings body) ->
  forallb (simple_usable rs) (evaluated_simples (state_expr (rs_state rs)))
  = true ->
  Q (plain_step (count_uses rs)).

Lemma region_step_forms : Q (region_step rs).
Proof using Q_undefined Q_begin Q_end Q_local Q_local_cell Q_let Q_static
  Q_region_apply_cont Q_plain.
  unfold region_step. cbv zeta.
  destruct
    (simples_fault rs (evaluated_simples (state_expr (rs_state rs)))
       (region_places rs))
    eqn:Hfault; [ apply Q_undefined | ].
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He;
    try (apply Q_plain;
         [ intros; discriminate | intros; discriminate
         | unfold region_places in Hfault; rewrite He in Hfault;
           exact (simples_fault_no_place rs _ Hfault) ]).
  - unfold region_places in Hfault. rewrite He in Hfault.
    assert (Hrest :
              forallb (simple_usable rs) (simples_of_named d) = true ->
              Q match marker_of x body with
                | Marker y region body' =>
                  local_allocation (count_uses rs) x y kind d region body'
                | Bad_marker => Region_undefined Not_of_the_conversion
                | No_marker =>
                  if (is_allocation d || is_cell_allocation d)
                     && existsb (simple_local (count_uses rs))
                          (simples_of_named d)
                  then Region_undefined Heap_holds_local
                  else
                    match store_fault (count_uses rs) d with
                    | Some fault => Region_undefined fault
                    | None => plain_step (count_uses rs)
                    end
                end).
    { intros Husable.
      destruct (marker_of x body) as [ | y region body' | ] eqn:Hmarker.
      - destruct
          ((is_allocation d || is_cell_allocation d)
           && existsb (simple_local (count_uses rs)) (simples_of_named d));
          [ apply Q_undefined | ].
        destruct (store_fault (count_uses rs) d); [ apply Q_undefined | ].
        exact (Q_let x kind d body eq_refl Husable).
      - destruct (is_cell_allocation d && kind_eqb kind Value) eqn:Hcell.
        + rewrite (local_allocation_cell _ _ _ _ _ _ _ Hcell).
          apply andb_true_iff in Hcell as [Hcell _].
          exact
            (Q_local_cell x kind d body y region body' eq_refl Hmarker
               Husable Hcell).
        + exact
            (Q_local x kind d body y region body' eq_refl Hmarker Husable
               Hcell).
      - apply Q_undefined. }
    destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
      try (apply Hrest; exact (simples_fault_no_place rs _ Hfault)).
    destruct rp, args as [ | arg [ | arg' args]];
      first
        [ apply Q_undefined
        | exact (Q_begin x kind _ _ body eq_refl)
        | exact (Q_end x kind _ _ arg body eq_refl) ].
  - unfold region_places in Hfault. rewrite He in Hfault.
    destruct
      (existsb (simple_local (count_uses rs))
         (evaluated_simples (Let_static bindings body)));
      [ apply Q_undefined | ].
    exact
      (Q_static bindings body eq_refl (simples_fault_no_place rs _ Hfault)).
  - destruct (count_uses_with_counts rs) as [counts Hcounts].
    rewrite Hcounts, existsb_open_mark_with_counts, <- Hcounts.
    destruct (existsb (simple_open_mark rs) (ac_args ac)) eqn:Hmark.
    + exact (Q_region_apply_cont ac eq_refl Hmark Hfault).
    + apply Q_plain; [ intros; discriminate | intros; discriminate | ].
      exact (simples_fault_no_mark rs _ _ Hfault Hmark).
Qed.

End Region_step_forms.

(* ======================================================================== *)
(* The frames stay in the order of their marks                              *)
(* ======================================================================== *)

(* This part is C2's. *)

Lemma frames_ordered_below_le (frames : list frame) (above above' : symbol) :
  (above <= above')%positive ->
  frames_ordered_below above frames = true ->
  frames_ordered_below above' frames = true.
Proof.
  destruct frames as [ | f below]; intros Hle; simpl; [ reflexivity | ].
  rewrite !andb_true_iff. intros [[Hm Hl] Hb].
  split; [ split | exact Hb ].
  - apply Pos.ltb_lt in Hm. apply Pos.ltb_lt. lia.
  - rewrite forallb_forall in *. intros [s a] Hin.
    specialize (Hl _ Hin). simpl in Hl.
    apply andb_true_iff in Hl as [H1 H2].
    apply andb_true_iff. split; [ exact H1 | ].
    apply Pos.ltb_lt in H2. apply Pos.ltb_lt. lia.
Qed.

(* What is below a frame of the stack is in order. *)
Lemma frames_ordered_below_app (above0 : list frame) :
  ∀ (above : symbol) (f : frame) (below : list frame),
    frames_ordered_below above (above0 ++ f :: below) = true ->
    frames_ordered_below above below = true.
Proof.
  induction above0 as [ | g above0 IH]; intros above f below; simpl.
  - rewrite !andb_true_iff. intros [[Hm _] Hb]. apply Pos.ltb_lt in Hm.
    apply (frames_ordered_below_le below (frame_mark f) above);
      [ lia | exact Hb ].
  - rewrite !andb_true_iff. intros [[Hm _] Hb]. apply Pos.ltb_lt in Hm.
    apply (frames_ordered_below_le below (frame_mark g) above); [ lia | ].
    exact (IH _ _ _ Hb).
Qed.

Lemma region_step_keeps_frames_ordered (rs rs' : region_state) :
  region_step rs = Region_step rs' ->
  frames_ordered rs = true ->
  frames_ordered rs' = true.
Proof.
  intros Hstep Hord. revert Hstep.
  assert (Hsame :
    ∀ rs0 : region_state,
      rs_frames rs0 = rs_frames rs -> rs_next rs0 = rs_next rs ->
      frames_ordered rs0 = true).
  { intros rs0 Hf Hn. unfold frames_ordered. rewrite Hf, Hn. exact Hord. }
  apply
    (region_step_cases rs
       (fun result => result = Region_step rs' -> frames_ordered rs' = true)).
  - discriminate.
  - intros x kind d mark_kind body _ Hstep.
    rewrite (begin_region_step _ _ _ _ _ _ Hstep).
    unfold frames_ordered in *. cbn.
    rewrite Hord, !andb_true_r. apply Pos.ltb_lt. lia.
  - intros x kind d ended_by region body _ Hstep.
    destruct (end_region_step _ _ _ _ _ _ _ Hstep)
      as (m & above & f & below & counts & _ & Hframes & ->).
    unfold frames_ordered in *. cbn in *. rewrite Hframes in Hord.
    exact (frames_ordered_below_app _ _ _ _ Hord).
  - intros x kind d body y region body' _ _ _ Hcell Hstep.
    destruct (local_allocation_step _ _ _ _ _ _ _ _ Hcell Hstep)
      as (v & sym & bindings & _ & _ & Hlocalise & _ & _ & _ & _ & Hnext & _
          & Hframes).
    destruct (localise_makes _ _ _ _ Hlocalise) as (_ & _ & Hmade).
    unfold frames_ordered in *. rewrite Hnext. cbn in *.
    pose proof (advance_le (rs_next rs) (length bindings)) as Hle.
    destruct (rs_frames rs) as [ | [m mark_kind locals] frames].
    + destruct Hframes as [-> _]. reflexivity.
    + destruct Hframes as [-> _]. cbn in *.
      rewrite !andb_true_iff in *. destruct Hord as [[Hm Hl] Hb].
      apply Pos.ltb_lt in Hm.
      split; [ split | exact Hb ].
      * apply Pos.ltb_lt. lia.
      * rewrite forallb_app. apply andb_true_iff. split.
        -- apply forallb_forall. intros [s a] Hin.
           destruct (Hmade _ _ Hin) as (Hfrom & Hto & _).
           apply andb_true_iff. split; apply Pos.ltb_lt; lia.
        -- rewrite forallb_forall in *. intros [s a] Hin.
           specialize (Hl _ Hin). simpl in Hl.
           apply andb_true_iff in Hl as [H1 H2].
           apply andb_true_iff. split; [ exact H1 | ].
           apply Pos.ltb_lt in H2. apply Pos.ltb_lt. lia.
  - intros x kind d body y region body' _ _ _ _ Hstep.
    destruct (local_cell_allocation_step _ _ _ _ _ _ _ Hstep)
      as (l & hp & rd & _ & _ & _ & Hframes & _ & Hnext & _).
    exact (Hsame rs' Hframes Hnext).
  - intros x kind d body _ _ _ _ Hstep.
    destruct (plain_step_step _ _ Hstep) as (s' & _ & ->).
    apply Hsame; reflexivity.
  - intros bindings body He _ _ Hstep.
    destruct (static_step_step (count_uses rs) rs' bindings body He Hstep)
      as (statics' & _ & ->).
    apply Hsame; reflexivity.
  - intros ac _ Hstep.
    destruct (region_apply_cont_step _ _ _ Hstep) as (Hf & _ & Hn & _).
    apply Hsame; assumption.
  - intros _ _ Hstep.
    destruct (plain_step_step _ _ Hstep) as (s' & _ & ->).
    apply Hsame; reflexivity.
Qed.

(* The state that a run starts in has no frame. *)
Lemma frames_ordered_initial (e : expr) (statics : static_env)
    (base : symbol) :
  frames_ordered (initial_region_state e statics base) = true.
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* Where a call, or the application of a continuation, lands                *)
(* ======================================================================== *)

(* A partial application that a call makes is one. *)
Lemma shape_of_call_partial (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (partial : value) :
  shape_of_call statics callee kind vs args_arity return_arity
  = Some (Partial_call partial) ->
  ∃ (closure : value) (args : list value) (arity : arity),
    partial = Value_partial_application closure args arity.
Proof.
  intros Hshape.
  assert (Hclosure :
    ∀ (closure : value) (kind : call_kind) (vs : list value)
      (args_arity : arity),
      shape_of_closure_call statics closure kind vs args_arity return_arity
      = Some (Partial_call partial) ->
      ∃ (closure : value) (args : list value) (arity : arity),
        partial = Value_partial_application closure args arity).
  { intros closure kind0 vs0 args_arity0 H.
    destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ H)
      as (called & slots & c & vs' & args_arity' & _ & _ & _ & _ & Hcall).
    destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
      as [[_ [=]] | [(_ & _ & _ & [=]) | (_ & _ & _ & _ & [= ->])]].
    eexists _, _, _. reflexivity. }
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [H | (closure & applied & applied_arity & _ & _ & _ & H)];
    exact (Hclosure _ _ _ _ H).
Qed.

(* [P] is asked of the value that a call calls, of its arguments, and
   of the arguments that remain of an over-application: of what code is
   given. [Pe] is asked of the values that a continuation is given: the
   arguments of an [Apply_cont], the extra arguments of an exception
   continuation, and a partial application that a call returns. [K] is
   asked of a continuation, and [Q] is what is shown of the variables
   and of the continuations of the state that a call, or the
   application of a continuation, lands on. *)
Section Call_chain.

Variable P : value -> Prop.
Variable Pe : value -> Prop.
Variable K : cont_value -> Prop.
Variable Q : env -> cont_env -> Prop.
Variable statics : static_env.

Hypothesis P_partial :
  ∀ (callee : value) (args : list value) (args_arity : arity),
    P (Value_partial_application callee args args_arity)
    <-> P callee ∧ Forall P args.
Hypothesis P_tuple : P_fields P statics.
Hypothesis Pe_P : ∀ v : value, Pe v -> P v.
Hypothesis Pe_partial :
  ∀ (callee : value) (args : list value) (args_arity : arity),
    P (Value_partial_application callee args args_arity) ->
    Pe (Value_partial_application callee args args_arity).
Hypothesis K_extra :
  ∀ (c : cont_value) (extra : list value),
    K (Extra_args c extra) <-> K c ∧ Forall Pe extra.
Hypothesis K_over :
  ∀ (remaining : list value) (remaining_arity return_arity : arity)
    (return_to exn_to : cont_value),
    K
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
    <-> Forall P remaining ∧ K return_to ∧ K exn_to.
Hypothesis Q_code :
  ∀ (closure : value) (c : code) (vs : list value) (callee_vars : env)
    (return_to exn_to : cont_value),
    P closure ->
    Forall P vs ->
    K return_to ->
    K exn_to ->
    bind_params (code_params c) vs [(code_my_closure c, closure)]
    = Some callee_vars ->
    Q callee_vars
      [(code_return_continuation c, return_to);
       (code_exn_continuation c, exn_to)].
Hypothesis Q_handler :
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (handler_vars : env),
    K
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    Forall Pe vs ->
    bind_params params vs def_vars = Some handler_vars ->
    Q handler_vars (rec_closures recursive def_vars def_conts ++ def_conts).

Definition call_result (result : step_result) : Prop :=
  match result with
  | Step s' => Q (state_vars s') (state_conts s')
  | Exit _ vs _ _ => Forall Pe vs
  | Undefined => True
  end.

Lemma call_code_result (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (hp : heap) (rd : read_record) :
  P closure ->
  Forall P vs ->
  K return_to ->
  K exn_to ->
  call_result
    (call_code closure c vs args_arity return_arity return_to exn_to statics
       hp rd).
Proof.
  intros Hclosure Hvs Hreturn Hexn. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ] eqn:Hbind; [ | exact I ].
  exact
    (Q_code closure c vs callee_vars return_to exn_to Hclosure Hvs Hreturn
       Hexn Hbind).
Qed.

Lemma run_call_result (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (hp : heap) (rd : read_record) :
  shape_values_all P shape ->
  (∀ partial : value,
    shape = Some (Partial_call partial) ->
    ∃ (closure : value) (args : list value) (arity : arity),
      partial = Value_partial_application closure args arity) ->
  K return_to ->
  K exn_to ->
  (∀ partial : value, Pe partial -> call_result (return_partial partial)) ->
  call_result
    (run_call return_partial shape return_arity return_to exn_to statics hp rd).
Proof.
  intros Hshape Hpartial Hreturn Hexn Hgiven.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hshape |- *.
  - destruct Hshape as [Hclosure Hvs].
    apply call_code_result; assumption.
  - destruct Hshape as (Hclosure & Hvs & Hremaining).
    apply call_code_result; try assumption.
    apply K_over. auto.
  - destruct (Hpartial partial eq_refl) as (closure & args & arity & ->).
    apply Hgiven, Pe_partial, Hshape.
  - exact I.
Qed.

Lemma apply_cont_value_result (c : cont_value) (hp : heap) (rd : read_record) :
  ∀ vs : list value,
    K c -> Forall Pe vs -> call_result (apply_cont_value c vs statics hp rd).
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs Hc Hvs.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | exact I ].
    exact (Q_handler _ _ _ _ _ _ vs handler_vars Hc Hvs Hbind).
  - exact Hvs.
  - simpl. apply K_extra in Hc as [Hc Hextra]. apply (IH (vs ++ extra) Hc).
    apply Forall_app. split; assumption.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    apply K_over in Hc as (Hremaining & Hreturn & Hexn).
    unfold call_function. apply run_call_result; try assumption.
    + apply (shape_of_call_values_all P P_partial);
        [ exact P_tuple | exact (Pe_P v (Forall_inv Hvs))
        | exact Hremaining ].
    + intros partial. apply shape_of_call_partial.
    + intros partial Hpartial. apply IHreturn; [ exact Hreturn | ].
      apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

Lemma call_function_result (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (hp : heap) (rd : read_record) :
  P callee ->
  Forall P vs ->
  K return_to ->
  K exn_to ->
  call_result
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd).
Proof.
  intros Hcallee Hvs Hreturn Hexn. unfold call_function.
  apply run_call_result; try assumption.
  - apply (shape_of_call_values_all P P_partial); assumption.
  - intros partial. apply shape_of_call_partial.
  - intros partial Hpartial. apply apply_cont_value_result;
      [ exact Hreturn | ].
    apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

(* A step of [step], at a state whose static environment is
   [statics]. *)
Lemma step_call_result (s : state) :
  state_statics s = statics ->
  Q (state_vars s) (state_conts s) ->
  (∀ k : continuation, K (resolve k (state_conts s))) ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
     (v : value) (hp' : heap) (rd' : read_record),
    state_expr s = Let x kind d body ->
    exec_named statics (state_heap s) (state_reads s) (state_vars s) d
    = Some (v, hp', rd') ->
    Q ((x, v) :: state_vars s) (state_conts s)) ->
  (∀ (k : continuation) (h : continuation_handler) (body : expr),
    state_expr s = Let_cont_non_recursive k h body ->
    Q (state_vars s)
      ((k, Closure h (state_vars s) (state_conts s) []) :: state_conts s)) ->
  (∀ (handlers : list (continuation * continuation_handler)) (body : expr),
    state_expr s = Let_cont_recursive handlers body ->
    Q (state_vars s)
      (rec_closures handlers (state_vars s) (state_conts s)
       ++ state_conts s)) ->
  (∀ (sim : simple) (v : value),
    In sim (evaluated_simples (state_expr s)) ->
    eval_simple (state_vars s) sim = Some v ->
    Pe v) ->
  call_result (step s).
Proof using K_extra K_over P_partial P_tuple Pe_P Pe_partial Q_code Q_handler.
  destruct s as [e vars conts statics0 hp rd].
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads].
  intros -> Hstate Hconts Hlet Hcont Hrec Hsimples.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step.
  - destruct (exec_named _ _ _ _ d) as [[[v hp'] rd'] | ] eqn:Hd;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    exact (Hlet x kind d body v hp' rd' eq_refl Hd).
  - destruct (eval_static_bindings vars bindings statics); [ | exact I ].
    exact Hstate.
  - exact (Hcont k h body eq_refl).
  - destruct (handlers_bind_statics handlers); [ exact I | ].
    exact (Hrec handlers body eq_refl).
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra; [ | exact I ].
    simpl in Hsimples.
    assert (Hextra_ok : Forall Pe extra).
    { apply (eval_simples_forall Pe vars (exn_extra_simples ap) extra);
        [ | exact Hextra ].
      intros sim w Hin. apply Hsimples. right. apply in_or_app. right.
      exact Hin. }
    apply call_function_result.
    + exact (Pe_P _ (Hsimples _ _ (or_introl eq_refl) Hcallee)).
    + apply (eval_simples_forall P vars (ap_args ap) vs); [ | exact Hargs ].
      intros sim w Hin Hw. apply Pe_P. apply (Hsimples sim w); [ | exact Hw ].
      right. apply in_or_app. left. exact Hin.
    + apply Hconts.
    + unfold with_extra_args. destruct extra as [ | w extra];
        [ apply Hconts | ].
      apply K_extra. split; [ apply Hconts | exact Hextra_ok ].
  - destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    unfold apply_continuation.
    apply apply_cont_value_result; [ apply Hconts | ].
    exact (eval_simples_forall Pe vars (ac_args ac) vs Hsimples Hargs).
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; [ | exact I ].
    exact Hstate.
  - exact I.
Qed.

End Call_chain.

(* ======================================================================== *)
(* A mark is held by variables only: the tests                              *)
(* ======================================================================== *)

Definition no_mark (frames : list frame) (v : value) : Prop :=
  value_no_mark frames v = true.

Lemma no_mark_partial (frames : list frame) (callee : value)
    (args : list value) (args_arity : arity) :
  no_mark frames (Value_partial_application callee args args_arity)
  <-> no_mark frames callee ∧ Forall (no_mark frames) args.
Proof.
  unfold no_mark. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity.
Qed.

Lemma value_no_mark_variable (frames : list frame) (v : value) :
  value_no_mark frames v = true -> variable_no_mark frames v = true.
Proof. destruct v; simpl; auto. Qed.

Lemma value_no_mark_parts (frames : list frame) (v w : value) :
  no_mark frames v -> In w (value_parts v) -> no_mark frames w.
Proof.
  unfold no_mark.
  destruct v; simpl; rewrite ?forallb_pairs; intros H Hin;
    first
      [ contradiction
      | exact (forallb_in _ _ w H Hin)
      | destruct Hin as [<- | []]; reflexivity ].
Qed.

Lemma value_no_mark_made (frames : list frame) (v : value) :
  is_made v = true ->
  forallb (value_no_mark frames) (value_parts v) = true ->
  value_no_mark frames v = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs; try discriminate; auto.
Qed.

Lemma allocated_no_mark_parts (frames : list frame)
    (allocated : statically_allocated_value) :
  allocated_no_mark frames allocated
  = forallb (value_no_mark frames) (allocated_values allocated)
    && forallb (fun s => negb (on_stack s frames))
         (allocated_symbols allocated).
Proof.
  destruct allocated; simpl; rewrite ?forallb_pairs, ?andb_true_r;
    try reflexivity.
  apply andb_comm.
Qed.

(* No value that the heap holds is a mark of [frames] or holds one:
   the last conjunct of FlambdaRegionsCallsSpec.marks_in_variables, as
   FlambdaHeapSpec.heap_values_all. *)
Definition heap_no_mark (frames : list frame) (hp : heap) : Prop :=
  heap_values_all (no_mark frames) hp.

Lemma heap_no_mark_forallb (frames : list frame) (hp : heap) :
  forallb (fun '(_, c) => forallb (value_no_mark frames) (values_of_cell c))
    hp
  = true
  <-> heap_no_mark frames hp.
Proof.
  unfold heap_no_mark, heap_values_all, no_mark. rewrite forallb_forall.
  split.
  - intros H l c v Hin Hv. specialize (H (l, c) Hin). cbv beta iota in H.
    exact (forallb_in _ _ v H Hv).
  - intros H [l c] Hin. apply forallb_forall. intros v Hv.
    exact (H l c v Hin Hv).
Qed.

Lemma marks_in_variables_parts (rs : region_state) :
  marks_in_variables rs = true
  <-> forallb (fun '(_, v) => variable_no_mark (rs_frames rs) v)
        (state_vars (rs_state rs))
      = true
      ∧ forallb (fun '(_, c) => cont_no_mark (rs_frames rs) c)
          (state_conts (rs_state rs))
        = true
      ∧ forallb
          (fun '(_, allocated) => allocated_no_mark (rs_frames rs) allocated)
          (static_values (state_statics (rs_state rs)))
        = true
      ∧ forallb
          (fun '(_, allocated) => allocated_no_mark (rs_frames rs) allocated)
          (live_locals (rs_frames rs))
        = true
      ∧ heap_no_mark (rs_frames rs) (state_heap (rs_state rs)).
Proof.
  unfold marks_in_variables. cbv zeta.
  rewrite !andb_true_iff, heap_no_mark_forallb. tauto.
Qed.

(* The tests under other frames: those of a value all of whose symbols
   are below [next], where the frames have no new mark below [next]. *)
Section No_mark_frames.

Variables frames frames' : list frame.
Variables base next : symbol.

Hypothesis no_new_mark :
  ∀ s : symbol,
    (s < next)%positive ->
    on_stack s frames = false ->
    on_stack s frames' = false.

Lemma value_no_mark_frames (v : value) :
  value_below next v = true ->
  value_no_mark frames v = true ->
  value_no_mark frames' v = true.
Proof.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    simpl; auto.
  - intros Hb Hn. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact
      (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) Hb Hin)
         (forallb_in _ _ (vslot, w) Hn Hin)).
  - intros Hb Hn. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact
      (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) Hb Hin)
         (forallb_in _ _ (vslot, w) Hn Hin)).
  - intros Hb Hn. unfold symbol_below in Hb. apply Pos.ltb_lt in Hb.
    apply negb_true_iff in Hn. apply negb_true_iff.
    exact (no_new_mark s Hb Hn).
  - intros Hb Hn. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH.
    exact (IH w Hin (forallb_in _ _ w Hb Hin) (forallb_in _ _ w Hn Hin)).
  - intros Hb Hn. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH.
    exact (IH w Hin (forallb_in _ _ w Hb Hin) (forallb_in _ _ w Hn Hin)).
  - intros Hb Hn. apply andb_true_iff in Hb as [Hbc Hba].
    apply andb_true_iff in Hn as [Hnc Hna].
    apply andb_true_iff. split; [ exact (IHcallee Hbc Hnc) | ].
    apply forallb_forall. intros arg Hin. rewrite Forall_forall in IHargs.
    exact
      (IHargs arg Hin (forallb_in _ _ arg Hba Hin)
         (forallb_in _ _ arg Hna Hin)).
Qed.

Lemma variable_no_mark_frames (v : value) :
  value_below next v = true ->
  variable_no_mark frames v = true ->
  variable_no_mark frames' v = true.
Proof.
  destruct v;
    first [ exact (value_no_mark_frames _) | intros _ _; reflexivity ].
Qed.

Lemma allocated_no_mark_frames (allocated : statically_allocated_value) :
  allocated_below next allocated = true ->
  allocated_no_mark frames allocated = true ->
  allocated_no_mark frames' allocated = true.
Proof.
  rewrite allocated_below_parts, !allocated_no_mark_parts.
  intros Hb Hn. apply andb_true_iff in Hb as [Hbv Hbs].
  apply andb_true_iff in Hn as [Hnv Hns]. apply andb_true_iff. split.
  - apply forallb_forall. intros w Hin.
    exact
      (value_no_mark_frames w (forallb_in _ _ w Hbv Hin)
         (forallb_in _ _ w Hnv Hin)).
  - apply forallb_forall. intros s Hin.
    pose proof (forallb_in _ _ s Hbs Hin) as Hb.
    pose proof (forallb_in _ _ s Hns Hin) as Hn. cbv beta in Hn.
    unfold symbol_below in Hb. apply Pos.ltb_lt in Hb.
    apply negb_true_iff in Hn. apply negb_true_iff.
    exact (no_new_mark s Hb Hn).
Qed.

Lemma cont_no_mark_frames (c : cont_value) :
  cont_below base next c = true ->
  cont_no_mark frames c = true ->
  cont_no_mark frames' c = true.
Proof.
  induction c
    as [h vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_deep_ind;
    simpl.
  - intros Hb Hn. apply andb_true_iff in Hb as [Hb _].
    apply andb_true_iff in Hb as [Hb Hbc].
    apply andb_true_iff in Hb as [_ Hbv].
    apply andb_true_iff in Hn as [Hnv Hnc]. apply andb_true_iff. split.
    + apply forallb_forall. intros [x v] Hin.
      exact
        (variable_no_mark_frames v (forallb_in _ _ (x, v) Hbv Hin)
           (forallb_in _ _ (x, v) Hnv Hin)).
    + apply forallb_forall. intros [k c] Hin. rewrite Forall_forall in IH.
      exact
        (IH (k, c) Hin (forallb_in _ _ (k, c) Hbc Hin)
           (forallb_in _ _ (k, c) Hnc Hin)).
  - reflexivity.
  - intros Hb Hn. apply andb_true_iff in Hb as [Hbc Hbe].
    apply andb_true_iff in Hn as [Hnc Hne].
    rewrite (IH Hbc Hnc). simpl. apply forallb_forall. intros w Hin.
    exact
      (value_no_mark_frames w (forallb_in _ _ w Hbe Hin)
         (forallb_in _ _ w Hne Hin)).
  - intros Hb Hn. apply andb_true_iff in Hb as [Hb Hbe].
    apply andb_true_iff in Hb as [Hbr Hbt].
    apply andb_true_iff in Hn as [Hn Hne].
    apply andb_true_iff in Hn as [Hnr Hnt].
    rewrite (IHreturn Hbt Hnt), (IHexn Hbe Hne), !andb_true_r.
    apply forallb_forall. intros w Hin.
    exact
      (value_no_mark_frames w (forallb_in _ _ w Hbr Hin)
         (forallb_in _ _ w Hnr Hin)).
Qed.

End No_mark_frames.

(* What [symbols_fresh] says of what a state holds and binds. *)
Lemma symbols_fresh_held (rs : region_state) :
  symbols_fresh rs = true ->
  forallb (fun '(_, v) => value_below (rs_next rs) v)
    (state_vars (rs_state rs))
  = true
  ∧ forallb (fun '(_, c) => cont_below (rs_base rs) (rs_next rs) c)
      (state_conts (rs_state rs))
    = true
  ∧ forallb
      (fun '(_, allocated) => allocated_below (rs_next rs) allocated)
      (static_values (state_statics (rs_state rs)))
    = true
  ∧ forallb
      (fun '(_, allocated) => allocated_below (rs_next rs) allocated)
      (live_locals (rs_frames rs))
    = true.
Proof.
  intros Hfresh. pose proof Hfresh as H. unfold symbols_fresh in H.
  cbv zeta in H. rewrite !andb_true_iff in H.
  destruct H as [[[[[[[[_ _] _] Hvars] Hconts] Hstatics] _] _] _].
  apply symbols_fresh_parts in Hfresh as (_ & _ & _ & _ & Hdrawn & _ & _).
  split; [ exact Hvars | ]. split; [ exact Hconts | ]. split.
  - revert Hstatics. apply forallb_impl. intros [s allocated] _ H.
    apply andb_true_iff in H as [_ H]. exact H.
  - apply forallb_forall. intros [s allocated] Hin. unfold live_locals in Hin.
    apply in_flat_map in Hin as (f & Hf & Hin).
    destruct (Hdrawn f Hf) as (_ & _ & Hlocals).
    exact (proj2 (proj2 (Hlocals s allocated Hin))).
Qed.

(* The mark of a frame was drawn, below the counter, and is bound to
   nothing: so a symbol that may be evaluated, one below the first that
   the machine drew, and one at or above the counter are the marks of
   no frame. *)
Lemma usable_not_on_stack (rs : region_state) (s : symbol) :
  symbols_fresh rs = true ->
  value_usable rs (Value_symbol s) = true ->
  on_stack s (rs_frames rs) = false.
Proof.
  intros Hfresh Hu.
  destruct (on_stack s (rs_frames rs)) eqn:Hon; [ | reflexivity ].
  unfold on_stack in Hon. apply existsb_exists in Hon as (f & Hin & Heq).
  apply Pos.eqb_eq in Heq. subst s.
  apply symbols_fresh_parts in Hfresh as (_ & _ & _ & _ & Hdrawn & Hmark & _).
  destruct (Hdrawn f Hin) as (Hle & _ & _).
  unfold value_usable, drawn, live_statics in Hu. cbn [static_values] in Hu.
  rewrite (Hmark f Hin) in Hu. simpl in Hu. rewrite orb_false_r in Hu.
  apply negb_true_iff in Hu. apply Pos.leb_gt in Hu. lia.
Qed.

Lemma not_drawn_not_on_stack (rs : region_state) (s : symbol) :
  symbols_fresh rs = true ->
  (s < rs_base rs)%positive ∨ (rs_next rs <= s)%positive ->
  on_stack s (rs_frames rs) = false.
Proof.
  intros Hfresh Hs.
  destruct (on_stack s (rs_frames rs)) eqn:Hon; [ | reflexivity ].
  unfold on_stack in Hon. apply existsb_exists in Hon as (f & Hin & Heq).
  apply Pos.eqb_eq in Heq. subst s.
  apply symbols_fresh_parts in Hfresh as (_ & _ & _ & _ & Hdrawn & _ & _).
  destruct (Hdrawn f Hin) as (Hle & Hlt & _). lia.
Qed.

(* Frames with no mark that those of the state have not. *)
Definition fewer_marks (rs : region_state) (frames' : list frame) : Prop :=
  ∀ s : symbol,
    on_stack s frames' = true -> on_stack s (rs_frames rs) = true.

Lemma fewer_marks_no_new (rs : region_state) (frames' : list frame) :
  fewer_marks rs frames' ->
  ∀ s : symbol,
    (s < rs_next rs)%positive ->
    on_stack s (rs_frames rs) = false ->
    on_stack s frames' = false.
Proof.
  intros Hfewer s _ Hoff.
  destruct (on_stack s frames') eqn:Hon; [ | reflexivity ].
  rewrite (Hfewer s Hon) in Hoff. discriminate.
Qed.

(* The five parts of the test of a state, under other frames. The
   values that the heap holds name no symbol at or above the counter:
   FlambdaRegionsInvariantSpec.cells_live asks it. *)
Lemma marks_frames (rs : region_state) (frames' : list frame) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  marks_in_variables rs = true ->
  (∀ s : symbol,
    (s < rs_next rs)%positive ->
    on_stack s (rs_frames rs) = false ->
    on_stack s frames' = false) ->
  forallb (fun '(_, v) => variable_no_mark frames' v)
    (state_vars (rs_state rs))
  = true
  ∧ forallb (fun '(_, c) => cont_no_mark frames' c)
      (state_conts (rs_state rs))
    = true
  ∧ forallb (fun '(_, allocated) => allocated_no_mark frames' allocated)
      (static_values (state_statics (rs_state rs)))
    = true
  ∧ forallb (fun '(_, allocated) => allocated_no_mark frames' allocated)
      (live_locals (rs_frames rs))
    = true
  ∧ heap_no_mark frames' (state_heap (rs_state rs)).
Proof.
  intros Hfresh Hbelow Hmarks Hnew.
  apply marks_in_variables_parts in Hmarks as (M1 & M2 & M3 & M4 & M5).
  destruct (symbols_fresh_held rs Hfresh) as (B1 & B2 & B3 & B4).
  split; [ | split; [ | split; [ | split ] ] ];
    [ apply forallb_forall .. | ].
  - intros [x v] Hin.
    exact
      (variable_no_mark_frames _ _ _ Hnew v (forallb_in _ _ (x, v) B1 Hin)
         (forallb_in _ _ (x, v) M1 Hin)).
  - intros [k c] Hin.
    exact
      (cont_no_mark_frames _ _ (rs_base rs) _ Hnew c
         (forallb_in _ _ (k, c) B2 Hin) (forallb_in _ _ (k, c) M2 Hin)).
  - intros [s allocated] Hin.
    exact
      (allocated_no_mark_frames _ _ _ Hnew allocated
         (forallb_in _ _ (s, allocated) B3 Hin)
         (forallb_in _ _ (s, allocated) M3 Hin)).
  - intros [s allocated] Hin.
    exact
      (allocated_no_mark_frames _ _ _ Hnew allocated
         (forallb_in _ _ (s, allocated) B4 Hin)
         (forallb_in _ _ (s, allocated) M4 Hin)).
  - intros l c w Hin Hw.
    exact
      (value_no_mark_frames _ _ _ Hnew w (Hbelow l c w Hin Hw)
         (M5 l c w Hin Hw)).
Qed.

(* ======================================================================== *)
(* The values that a step reads hold no mark                                *)
(* ======================================================================== *)

(* The value of a simple is what a variable may hold. *)
Lemma eval_simple_variable_no_mark (frames : list frame) (vars : env)
    (sim : simple) (v : value) :
  forallb (fun '(_, v) => variable_no_mark frames v) vars = true ->
  eval_simple vars sim = Some v ->
  variable_no_mark frames v = true.
Proof.
  intros Hvars. destruct sim as [x | sym | c]; simpl.
  - exact (lookup_forallb (variable_no_mark frames) x v vars Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

(* The value of a simple that may be evaluated holds no mark. *)
Lemma simple_no_mark (rs : region_state) (frames' : list frame)
    (sim : simple) (v : value) :
  symbols_fresh rs = true ->
  fewer_marks rs frames' ->
  forallb (fun '(_, v) => variable_no_mark frames' v)
    (state_vars (rs_state rs))
  = true ->
  simple_usable rs sim = true ->
  eval_simple (state_vars (rs_state rs)) sim = Some v ->
  value_no_mark frames' v = true.
Proof.
  intros Hfresh Hfewer Hvars Hu Hv. unfold simple_usable in Hu.
  rewrite Hv in Hu.
  pose proof (eval_simple_variable_no_mark frames' _ sim v Hvars Hv) as Hvar.
  destruct v; try exact Hvar.
  simpl. apply negb_true_iff.
  destruct (on_stack s frames') eqn:Hon; [ | reflexivity ].
  pose proof (Hfewer s Hon) as H.
  rewrite (usable_not_on_stack rs s Hfresh Hu) in H. discriminate H.
Qed.

Lemma fewer_marks_same (rs : region_state) : fewer_marks rs (rs_frames rs).
Proof. intros s H. exact H. Qed.

(* What a symbol is bound to, in a frame or in the static environment
   of the state, holds no mark. *)
Lemma no_mark_static (rs : region_state) (s : symbol)
    (allocated : statically_allocated_value) :
  marks_in_variables rs = true ->
  no_mark (rs_frames rs) (Value_symbol s) ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  (∀ w : value,
    In w (allocated_values allocated) -> no_mark (rs_frames rs) w)
  ∧ (∀ s' : symbol,
      In s' (allocated_symbols allocated) ->
      no_mark (rs_frames rs) (Value_symbol s')).
Proof.
  intros Hmarks _ Hlookup.
  apply marks_in_variables_parts in Hmarks as (_ & _ & M3 & M4 & _).
  assert (Hallocated : allocated_no_mark (rs_frames rs) allocated = true).
  { apply lookup_in in Hlookup. unfold live_statics in Hlookup.
    cbn [static_values] in Hlookup.
    apply in_app_iff in Hlookup as [Hin | Hin];
      [ exact (forallb_in _ _ (s, allocated) M4 Hin)
      | exact (forallb_in _ _ (s, allocated) M3 Hin) ]. }
  rewrite allocated_no_mark_parts in Hallocated.
  apply andb_true_iff in Hallocated as [Hv Hs]. split.
  - intros w Hin. exact (forallb_in _ _ w Hv Hin).
  - intros s' Hin. exact (forallb_in _ _ s' Hs Hin).
Qed.

Lemma no_mark_fields (rs : region_state) :
  marks_in_variables rs = true ->
  P_fields (no_mark (rs_frames rs)) (live_statics rs).
Proof.
  intros Hmarks.
  apply (reads_fields (no_mark (rs_frames rs)) (live_statics rs)).
  - exact (value_no_mark_parts (rs_frames rs)).
  - intros s allocated. exact (no_mark_static rs s allocated Hmarks).
Qed.

(* The value of a [Let] whose simples may be evaluated holds no mark. *)
Lemma eval_named_no_mark (rs : region_state) (d : named) (v : value) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  eval_named (live_statics rs) (state_vars (rs_state rs)) d = Some v ->
  value_no_mark (rs_frames rs) v = true.
Proof.
  intros Hfresh Hmarks Husable Heval.
  pose proof Hmarks as HM.
  apply marks_in_variables_parts in HM as (M1 & _ & _ & _).
  assert (Hsimples :
    ∀ (sim : simple) (w : value),
      In sim (simples_of_named d) ->
      eval_simple (state_vars (rs_state rs)) sim = Some w ->
      no_mark (rs_frames rs) w).
  { intros sim w Hin.
    exact
      (simple_no_mark rs _ sim w Hfresh (fewer_marks_same rs) M1
         (forallb_in _ _ sim Husable Hin)). }
  destruct (is_allocation d) eqn:His.
  - destruct (eval_named_makes _ _ d v His Heval) as [Hmade Hparts].
    apply value_no_mark_made; [ exact Hmade | ].
    apply forallb_forall. intros w Hin.
    destruct (Hparts w Hin) as (sim & Hsim & Hw).
    exact (Hsimples sim w Hsim Hw).
  - exact
      (eval_named_reads (no_mark (rs_frames rs)) (live_statics rs)
         (fun c _ => eq_refl) (value_no_mark_parts (rs_frames rs))
         (fun slot slot' decls slots H => H)
         (fun s allocated => no_mark_static rs s allocated Hmarks)
         (state_vars (rs_state rs)) d v His Hsimples Heval).
Qed.

(* The same of a [Let] that is run on the heap: its value holds no
   mark, and neither does a value that the heap holds after it. A
   primitive makes its value, and the cells that it makes or changes,
   out of its arguments and of the cells that it reads
   (FlambdaSemanticsSpec.exec_named_preserves_statement). *)
Lemma exec_named_no_mark (rs : region_state) (d : named) (v : value)
    (hp' : heap) (rd' : read_record) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  exec_named (live_statics rs) (state_heap (rs_state rs))
    (state_reads (rs_state rs)) (state_vars (rs_state rs)) d
  = Some (v, hp', rd') ->
  value_no_mark (rs_frames rs) v = true ∧ heap_no_mark (rs_frames rs) hp'.
Proof.
  intros Hfresh Hmarks Husable Hexec.
  pose proof Hmarks as HM.
  apply marks_in_variables_parts in HM as (M1 & _ & _ & _ & M5).
  assert (Hcases :
    (∃ (p : prim) (args : list simple), d = Prim p args)
    ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d;
      first [ left; eexists _, _; reflexivity | right; intros; discriminate ]. }
  destruct Hcases as [(p & args & ->) | Hnot].
  - apply
      (exec_named_preserves (no_mark (rs_frames rs)) (live_statics rs)
         (state_heap (rs_state rs)) hp' (state_reads (rs_state rs)) rd'
         (state_vars (rs_state rs)) p args v);
      try assumption.
    + intros c. split; reflexivity.
    + intros tag fields. unfold no_mark. simpl. apply forallb_forall.
    + intros elements. unfold no_mark. simpl. apply forallb_forall.
    + intros s tag fields Hs Hlookup.
      exact (proj1 (no_mark_static rs s _ Hmarks Hs Hlookup)).
    + intros s elements Hs Hlookup.
      exact (proj1 (no_mark_static rs s _ Hmarks Hs Hlookup)).
    + intros l. reflexivity.
    + intros vs Hvs arg Harg.
      refine
        (proj1 (Forall_forall _ vs)
           (eval_simples_forall _ (state_vars (rs_state rs)) args vs _ Hvs)
           arg Harg).
      intros sim w Hin.
      exact
        (simple_no_mark rs _ sim w Hfresh (fewer_marks_same rs) M1
           (forallb_in _ _ sim Husable Hin)).
  - rewrite (exec_named_not_prim _ _ _ _ d Hnot) in Hexec.
    destruct (eval_named (live_statics rs) (state_vars (rs_state rs)) d)
      as [v0 | ] eqn:Hnamed; [ | discriminate ].
    simpl in Hexec. injection Hexec as -> <- _.
    split; [ | exact M5 ].
    exact (eval_named_no_mark rs d v Hfresh Hmarks Husable Hnamed).
Qed.

(* What is asked below of a step of [step], of a [Let] whose simples
   may be evaluated. *)
Lemma let_usable_no_mark (rs : region_state) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  state_expr (rs_state rs) = Let x kind d body ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  forallb (simple_usable rs) (evaluated_simples (state_expr (rs_state rs)))
  = true
  ∧ ∀ (x' : variable) (kind' : flambda_kind) (d' : named) (body' : expr)
      (v : value) (hp' : heap) (rd' : read_record),
      state_expr (rs_state rs) = Let x' kind' d' body' ->
      exec_named (live_statics rs) (state_heap (rs_state rs))
        (state_reads (rs_state rs)) (state_vars (rs_state rs)) d'
      = Some (v, hp', rd') ->
      value_no_mark (rs_frames rs) v = true
      ∧ heap_no_mark (rs_frames rs) hp'.
Proof.
  intros Hfresh Hmarks He Husable. split.
  - rewrite He. apply forallb_forall. intros sim Hin.
    exact
      (forallb_in _ _ sim Husable
         (evaluated_simples_let x kind d body sim Hin)).
  - intros x' kind' d' body' v hp' rd' He' Hexec. rewrite He in He'.
    injection He' as <- <- <- <-.
    exact (exec_named_no_mark rs d v hp' rd' Hfresh Hmarks Husable Hexec).
Qed.

(* ======================================================================== *)
(* A step keeps "a mark is held by variables only"                          *)
(* ======================================================================== *)

(* The test of the variables and of the continuations of a state. *)
Definition marks_held (frames : list frame) (vars : env) (conts : cont_env)
    : Prop :=
  forallb (fun '(_, v) => variable_no_mark frames v) vars = true
  ∧ forallb (fun '(_, c) => cont_no_mark frames c) conts = true.

Lemma bind_params_marks (frames : list frame) (params : list variable)
    (vs : list value) (vars vars' : env) :
  Forall (fun v => variable_no_mark frames v = true) vs ->
  forallb (fun '(_, v) => variable_no_mark frames v) vars = true ->
  bind_params params vs vars = Some vars' ->
  forallb (fun '(_, v) => variable_no_mark frames v) vars' = true.
Proof.
  intros Hvs Hvars Hbind. apply forallb_pairs_Forall.
  apply
    (bind_params_values_all (fun v => variable_no_mark frames v = true)
       params vs vars vars' Hvs);
    [ | exact Hbind ].
  apply forallb_pairs_Forall. exact Hvars.
Qed.

Lemma rec_closures_no_mark (frames : list frame)
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  forallb (fun '(_, v) => variable_no_mark frames v) vars = true ->
  forallb (fun '(_, c) => cont_no_mark frames c) conts = true ->
  forallb (fun '(_, c) => cont_no_mark frames c)
    (rec_closures handlers vars conts)
  = true.
Proof.
  intros Hv Hc. unfold rec_closures. generalize handlers at 1 as all.
  intros all.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite Hv, Hc, IH. reflexivity.
Qed.

Lemma resolve_no_mark (frames : list frame) (k : continuation)
    (conts : cont_env) :
  forallb (fun '(_, c) => cont_no_mark frames c) conts = true ->
  cont_no_mark frames (resolve k conts) = true.
Proof.
  intros Hconts. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hlookup; [ | reflexivity ].
  exact (lookup_forallb (cont_no_mark frames) k c conts Hconts Hlookup).
Qed.

Lemma no_mark_extra (frames : list frame) (c : cont_value)
    (extra : list value) :
  cont_no_mark frames (Extra_args c extra) = true
  <-> cont_no_mark frames c = true ∧ Forall (no_mark frames) extra.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

Lemma no_mark_over (frames : list frame) (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_no_mark frames
    (Over_application remaining remaining_arity return_arity return_to exn_to)
  = true
  <-> Forall (no_mark frames) remaining
      ∧ cont_no_mark frames return_to = true
      ∧ cont_no_mark frames exn_to = true.
Proof. simpl. rewrite !andb_true_iff, forallb_Forall. tauto. Qed.

Lemma marks_code (frames : list frame) (closure : value) (c : code)
    (vs : list value) (callee_vars : env) (return_to exn_to : cont_value) :
  no_mark frames closure ->
  Forall (no_mark frames) vs ->
  cont_no_mark frames return_to = true ->
  cont_no_mark frames exn_to = true ->
  bind_params (code_params c) vs [(code_my_closure c, closure)]
  = Some callee_vars ->
  marks_held frames callee_vars
    [(code_return_continuation c, return_to);
     (code_exn_continuation c, exn_to)].
Proof.
  intros Hclosure Hvs Hreturn Hexn Hbind. split.
  - apply
      (bind_params_marks frames (code_params c) vs
         [(code_my_closure c, closure)] callee_vars);
      [ | | exact Hbind ].
    + exact (Forall_impl _ (value_no_mark_variable frames) Hvs).
    + simpl. rewrite (value_no_mark_variable frames closure Hclosure).
      reflexivity.
  - simpl. rewrite Hreturn, Hexn. reflexivity.
Qed.

Lemma marks_handler (frames : list frame) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (def_vars : env)
    (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (handler_vars : env) :
  cont_no_mark frames
    (Closure (Handler params metadata handler) def_vars def_conts recursive)
  = true ->
  Forall (fun v => variable_no_mark frames v = true) vs ->
  bind_params params vs def_vars = Some handler_vars ->
  marks_held frames handler_vars
    (rec_closures recursive def_vars def_conts ++ def_conts).
Proof.
  simpl. intros Hc Hvs Hbind.
  apply andb_true_iff in Hc as [Hdef_vars Hdef_conts]. split.
  - exact
      (bind_params_marks frames params vs def_vars handler_vars Hvs Hdef_vars
         Hbind).
  - rewrite forallb_app,
      (rec_closures_no_mark frames recursive def_vars def_conts Hdef_vars
         Hdef_conts),
      Hdef_conts.
    reflexivity.
Qed.

(* What is shown of a step. *)
Definition marks_step (result : region_step_result) : Prop :=
  match result with
  | Region_step rs' => marks_in_variables rs' = true
  | Region_exit _ _ _ | Region_undefined _ => True
  end.

Lemma plain_step_marks (rs : region_state) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  forallb (simple_usable rs) (evaluated_simples (state_expr (rs_state rs)))
  = true ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
     (v : value) (hp' : heap) (rd' : read_record),
    state_expr (rs_state rs) = Let x kind d body ->
    exec_named (live_statics rs) (state_heap (rs_state rs))
      (state_reads (rs_state rs)) (state_vars (rs_state rs)) d
    = Some (v, hp', rd') ->
    value_no_mark (rs_frames rs) v = true
    ∧ heap_no_mark (rs_frames rs) hp') ->
  marks_step (plain_step rs).
Proof.
  intros Hfresh Hmarks Husable Hlet.
  pose proof Hmarks as HM.
  apply marks_in_variables_parts in HM as (M1 & M2 & M3 & M4 & M5).
  assert (Hstep :
    call_result (no_mark (rs_frames rs)) (marks_held (rs_frames rs))
      (step (with_statics (rs_state rs) (live_statics rs)))).
  { apply
      (step_call_result (no_mark (rs_frames rs)) (no_mark (rs_frames rs))
         (fun c => cont_no_mark (rs_frames rs) c = true)
         (marks_held (rs_frames rs)) (live_statics rs));
      cbn [with_statics state_expr state_vars state_conts state_statics
           state_heap state_reads].
    - exact (no_mark_partial _).
    - exact (no_mark_fields rs Hmarks).
    - intros v H. exact H.
    - intros callee args args_arity H. exact H.
    - exact (no_mark_extra _).
    - exact (no_mark_over _).
    - exact (marks_code _).
    - intros params metadata handler def_vars def_conts recursive vs
        handler_vars Hc Hvs.
      exact
        (marks_handler _ params metadata handler def_vars def_conts recursive
           vs handler_vars Hc
           (Forall_impl _ (value_no_mark_variable _) Hvs)).
    - reflexivity.
    - exact (conj M1 M2).
    - intros k. exact (resolve_no_mark _ k _ M2).
    - intros x kind d body v hp' rd' He Hd. split; [ | exact M2 ]. simpl.
      rewrite
        (value_no_mark_variable _ v
           (proj1 (Hlet x kind d body v hp' rd' He Hd))).
      exact M1.
    - intros k h body _. split; [ exact M1 | ]. simpl. rewrite M1, M2.
      reflexivity.
    - intros handlers body _. split; [ exact M1 | ].
      rewrite forallb_app, (rec_closures_no_mark _ handlers _ _ M1 M2), M2.
      reflexivity.
    - intros sim v Hin.
      exact
        (simple_no_mark rs _ sim v Hfresh (fewer_marks_same rs) M1
           (forallb_in _ _ sim Husable Hin)). }
  unfold plain_step.
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k vs hp rd | ] eqn:Hs; [ | exact I | exact I ].
  destruct Hstep as [Hv Hc]. apply marks_in_variables_parts.
  refine (conj Hv (conj Hc (conj M3 (conj M4 _)))).
  cbn [rs_state rs_frames with_statics state_heap].
  destruct (step_heap _ _ Hs)
    as [-> | (x & kind & d & body & v & He & Hexec)]; [ exact M5 | ].
  exact (proj2 (Hlet x kind d body v _ _ He Hexec)).
Qed.

(* An [Apply_cont] whose continuation is not bound to a handler: no
   argument is at the place of a region. *)
Lemma apply_cont_no_handler_usable (rs : region_state) (ac : apply_cont) :
  state_expr (rs_state rs) = Apply_cont ac ->
  simples_fault rs (ac_args ac) (region_places rs) = None ->
  (∀ (h : continuation_handler) (vars : env) (conts : cont_env)
     (recursive : list (continuation * continuation_handler)),
    resolve (ac_continuation ac) (state_conts (rs_state rs))
    <> Closure h vars conts recursive) ->
  forallb (simple_usable rs) (ac_args ac) = true.
Proof.
  intros He Hfault Hnot. apply simples_fault_no_place.
  unfold region_places in Hfault. rewrite He in Hfault.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try exact Hfault.
  contradiction (Hnot h vars conts recursive eq_refl).
Qed.

Lemma region_apply_cont_marks (rs : region_state) (ac : apply_cont) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  state_expr (rs_state rs) = Apply_cont ac ->
  ((∀ (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)),
     resolve (ac_continuation ac) (state_conts (rs_state rs))
     <> Closure h vars conts recursive) ->
   forallb (simple_usable rs) (ac_args ac) = true) ->
  marks_step (region_apply_cont rs ac).
Proof.
  intros Hfresh Hmarks He Husable. unfold region_apply_cont.
  pose proof Hmarks as HM.
  apply marks_in_variables_parts in HM as (M1 & M2 & M3 & M4 & M5).
  pose proof (resolve_no_mark (rs_frames rs) (ac_continuation ac) _ M2) as Hc.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to]
    eqn:Hresolve;
    try (apply (plain_step_marks rs Hfresh Hmarks);
         [ rewrite He; apply Husable; intros; discriminate
         | intros x kind d body v hp' rd' Hlet; rewrite He in Hlet;
           discriminate ]).
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ]
    eqn:Hargs; [ | exact I ].
  destruct (kinds_eqb _ _); [ | exact I ].
  destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
    [ | exact I ].
  assert (Hvs : Forall (fun v => variable_no_mark (rs_frames rs) v = true) vs).
  { apply
      (eval_simples_forall _ (state_vars (rs_state rs)) (ac_args ac) vs);
      [ | exact Hargs ].
    intros sim w _. exact (eval_simple_variable_no_mark _ _ sim w M1). }
  destruct
    (marks_handler _ params metadata handler def_vars def_conts recursive vs
       handler_vars Hc Hvs Hbind)
    as [Hv Hconts].
  apply marks_in_variables_parts.
  exact (conj Hv (conj Hconts (conj M3 (conj M4 M5)))).
Qed.

Lemma begin_region_marks (rs : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  marks_in_variables rs = true ->
  marks_step (begin_region rs x kind mark_kind body).
Proof.
  intros Hfresh Hbelow Hmarks. unfold begin_region.
  destruct (kind_eqb kind Region); [ | exact I ].
  destruct
    (marks_frames rs (Frame (rs_next rs) mark_kind [] :: rs_frames rs) Hfresh
       Hbelow Hmarks)
    as (N1 & N2 & N3 & N4 & N5).
  { intros s Hlt Hoff.
    change
      (Pos.eqb (rs_next rs) s || on_stack s (rs_frames rs) = false).
    rewrite Hoff, orb_false_r. apply Pos.eqb_neq. lia. }
  apply marks_in_variables_parts.
  exact (conj N1 (conj N2 (conj N3 (conj N4 N5)))).
Qed.

Lemma end_region_marks (rs : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  marks_in_variables rs = true ->
  marks_step (end_region rs x kind ended_by region body).
Proof.
  intros Hfresh Hbelow Hmarks.
  destruct (end_region rs x kind ended_by region body)
    as [rs' | k vs rs' | fault] eqn:Hstep; [ | exact I | exact I ].
  destruct (end_region_step _ _ _ _ _ _ _ Hstep)
    as (m & above & f & below & counts & _ & Hframes & ->).
  destruct (marks_frames rs below Hfresh Hbelow Hmarks)
    as (N1 & N2 & N3 & N4 & N5).
  { apply fewer_marks_no_new. intros s Hon. rewrite Hframes, on_stack_app.
    change (on_stack s (f :: below))
      with (Pos.eqb (frame_mark f) s || on_stack s below).
    rewrite Hon, !orb_true_r. reflexivity. }
  apply marks_in_variables_parts.
  refine (conj N1 (conj N2 (conj N3 (conj _ N5)))).
  rewrite Hframes, live_locals_app, forallb_app in N4.
  apply andb_true_iff in N4 as [_ N4].
  change (live_locals (f :: below))
    with (frame_locals f ++ live_locals below) in N4.
  rewrite forallb_app in N4. apply andb_true_iff in N4 as [_ N4]. exact N4.
Qed.

Lemma local_allocation_marks (rs : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  marks_in_variables rs = true ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  is_cell_allocation d && kind_eqb kind Value = false ->
  marks_step (local_allocation rs x y kind d region body).
Proof.
  intros Hfresh Hbelow Hmarks Husable Hcell.
  destruct (local_allocation rs x y kind d region body)
    as [rs' | k vs rs' | fault] eqn:Hstep; [ | exact I | exact I ].
  destruct (local_allocation_step rs rs' x y kind d region body Hcell Hstep)
    as (v & sym & bindings & His & Heval & Hlocalise & _ & Hvars & Hconts
        & _ & _ & _ & Hframes).
  destruct (local_allocation_memory rs rs' x y kind d region body Hcell Hstep)
    as (Hheap & _ & _).
  destruct (eval_named_makes _ _ d v His Heval) as [_ Hparts].
  destruct (localise_makes _ _ _ _ Hlocalise) as (_ & _ & Hbound).
  assert (Hfewer : fewer_marks rs (rs_frames rs')).
  { unfold fewer_marks.
    destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
      destruct Hframes as [-> _]; intros s H; exact H. }
  destruct
    (marks_frames rs (rs_frames rs') Hfresh Hbelow Hmarks
       (fewer_marks_no_new rs _ Hfewer))
    as (N1 & N2 & N3 & N4 & N5).
  assert (Nb :
    forallb
      (fun '(_, allocated) => allocated_no_mark (rs_frames rs') allocated)
      bindings
    = true).
  { apply forallb_forall. intros [s allocated] Hin.
    destruct (Hbound s allocated Hin) as (_ & _ & Hvalues & Hsymbols).
    rewrite allocated_no_mark_parts, Hvalues. apply andb_true_iff. split.
    - apply forallb_forall. intros w Hw.
      destruct (Hparts w Hw) as (sim & Hsim & Hsim_w).
      exact
        (simple_no_mark rs _ sim w Hfresh Hfewer N1
           (forallb_in _ _ sim Husable Hsim) Hsim_w).
    - apply forallb_forall. intros s' Hs'.
      destruct (Hsymbols s' Hs') as (allocated' & Hin').
      destruct (Hbound s' allocated' Hin') as (Hfrom & _).
      apply negb_true_iff.
      destruct (on_stack s' (rs_frames rs')) eqn:Hon; [ | reflexivity ].
      pose proof (Hfewer s' Hon) as H.
      rewrite (not_drawn_not_on_stack rs s' Hfresh (or_intror Hfrom)) in H.
      discriminate H. }
  apply marks_in_variables_parts. rewrite Hvars, Hconts, Hheap.
  split; [ exact N1 | ]. split; [ exact N2 | ].
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames].
  - destruct Hframes as [Hframes' Hvalues'].
    rewrite Hvalues', forallb_app, Nb, N3, Hframes'. rewrite Hframes' in N5.
    split; [ reflexivity | ]. split; [ reflexivity | exact N5 ].
  - destruct Hframes as [Hframes' Hstatics']. rewrite Hstatics'.
    split; [ exact N3 | ]. split; [ | exact N5 ].
    rewrite Hframes' in Nb, N4 |- *.
    change
      (live_locals (Frame m mark_kind (bindings ++ locals) :: frames))
      with ((bindings ++ locals) ++ live_locals frames).
    rewrite <- app_assoc, forallb_app, Nb. exact N4.
Qed.

(* A local allocation of a cell: the frames are those of the state. *)
Lemma local_cell_allocation_marks (rs : region_state) (x y : variable)
    (d : named) (region : option simple) (body : expr) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  marks_step (local_cell_allocation rs x y d region body).
Proof.
  intros Hfresh Hmarks Husable.
  destruct (local_cell_allocation rs x y d region body)
    as [rs' | k vs rs' | fault] eqn:Hstep; [ | exact I | exact I ].
  destruct (local_cell_allocation_step rs rs' x y d region body Hstep)
    as (l & hp & rd & _ & Hexec & Hstate & Hframes & _ & _ & _).
  pose proof Hmarks as HM.
  apply marks_in_variables_parts in HM as (M1 & M2 & M3 & M4 & _).
  destruct (exec_named_no_mark rs d _ hp rd Hfresh Hmarks Husable Hexec)
    as [_ N5].
  apply marks_in_variables_parts. rewrite Hstate, Hframes.
  cbn [state_vars state_conts state_statics state_heap].
  refine (conj _ (conj M2 (conj M3 (conj M4 N5)))).
  simpl. exact M1.
Qed.

Lemma static_step_marks (rs : region_state) (bindings : list static_binding)
    (body : expr) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  state_expr (rs_state rs) = Let_static bindings body ->
  forallb (simple_usable rs) (flat_map simples_of_binding bindings) = true ->
  marks_step (static_step rs).
Proof.
  intros Hfresh Hmarks He Husable.
  destruct (static_step rs) as [rs' | k vs rs' | fault] eqn:Hstep;
    [ | exact I | exact I ].
  destruct (static_step_step rs rs' bindings body He Hstep)
    as (statics' & Heval & ->).
  destruct (eval_static_bindings_parts _ _ _ _ Heval)
    as (new_code & new_values & _ & Hvalues & _ & Hnew).
  pose proof Hmarks as HM.
  apply marks_in_variables_parts in HM as (M1 & M2 & M3 & M4 & M5).
  pose proof (symbols_fresh_expr rs Hfresh) as Hbelow.
  rewrite He in Hbelow. simpl in Hbelow.
  apply andb_true_iff in Hbelow as [Hbindings _].
  apply marks_in_variables_parts.
  cbn [rs_state rs_frames state_vars state_conts state_statics state_heap].
  refine (conj M1 (conj M2 (conj _ (conj M4 M5)))).
  rewrite Hvalues, forallb_app, M3, andb_true_r.
  apply forallb_forall. intros [s allocated] Hin.
  destruct (Hnew s allocated Hin) as (const & Hconst_in & Hconst).
  destruct (eval_static_const_makes _ const allocated Hconst)
    as [Hparts Hsymbols].
  pose proof (forallb_in _ _ _ Hbindings Hconst_in) as Hbinding.
  simpl in Hbinding. apply andb_true_iff in Hbinding as [_ Hconst_below].
  unfold static_const_below in Hconst_below.
  apply andb_true_iff in Hconst_below as [_ Hset_below].
  rewrite allocated_no_mark_parts. apply andb_true_iff. split.
  - apply forallb_forall. intros w Hw.
    destruct (Hparts w Hw) as (sim & Hsim & Hsim_w).
    assert (Hsim_in : In sim (flat_map simples_of_binding bindings))
      by (apply in_flat_map; exists (Bind_symbol s const);
          split; [ exact Hconst_in | exact Hsim ]).
    exact
      (simple_no_mark rs _ sim w Hfresh (fewer_marks_same rs) M1
         (forallb_in _ _ sim Husable Hsim_in) Hsim_w).
  - rewrite Hsymbols.
    destruct const
      as [code value_slots | slot closure_symbols function_decls value_slots
         | tag fields | contents | bytes | elements];
      try reflexivity.
    rewrite forallb_pairs in Hset_below.
    revert Hset_below. apply forallb_impl. intros s' _ Hs'.
    apply negb_true_iff. apply (not_drawn_not_on_stack rs s' Hfresh). left.
    unfold symbol_below in Hs'. apply Pos.ltb_lt in Hs'. exact Hs'.
Qed.

(* A step of a state that has the invariant keeps the first part of the
   test. *)
Theorem region_step_marks (rs : region_state) :
  symbols_fresh rs = true ->
  cells_below (rs_next rs) (state_heap (rs_state rs)) ->
  marks_in_variables rs = true ->
  marks_step (region_step rs).
Proof.
  intros Hfresh Hbelow Hmarks. apply region_step_forms.
  - intros fault. exact I.
  - intros x kind d mark_kind body _.
    exact (begin_region_marks (count_uses rs) x kind mark_kind body Hfresh
             Hbelow Hmarks).
  - intros x kind p ended_by region body _.
    exact (end_region_marks (count_uses rs) x kind ended_by region body
             Hfresh Hbelow Hmarks).
  - intros x kind d body y region body' _ _ Husable Hcell.
    exact (local_allocation_marks (count_uses rs) x y kind d region body'
             Hfresh Hbelow Hmarks Husable Hcell).
  - intros x kind d body y region body' _ _ Husable _.
    exact (local_cell_allocation_marks (count_uses rs) x y d region body'
             Hfresh Hmarks Husable).
  - intros x kind d body He Husable.
    destruct
      (let_usable_no_mark (count_uses rs) x kind d body Hfresh Hmarks He
         Husable)
      as [Hsimples Hlet].
    exact (plain_step_marks (count_uses rs) Hfresh Hmarks Hsimples Hlet).
  - intros bindings body He Husable.
    exact (static_step_marks (count_uses rs) bindings body Hfresh Hmarks He
             Husable).
  - intros ac He Hmark Hfault.
    apply (region_apply_cont_marks (count_uses rs) ac Hfresh Hmarks He).
    exact (apply_cont_no_handler_usable rs ac He Hfault).
  - intros Hnot_let _ Husable.
    apply (plain_step_marks (count_uses rs) Hfresh Hmarks Husable).
    intros x kind d body v hp' rd' He.
    contradiction (Hnot_let x kind d body He).
Qed.

(* ======================================================================== *)
(* What the continuations have held: the tests                              *)
(* ======================================================================== *)

(* The test of [held_older], of the value of one variable. *)
Definition older (frames : list frame) (cells : list (location * mark))
    (vs : list value) (w : value) : bool :=
  match w with
  | Value_symbol m =>
    negb (on_stack m frames) || forallb (held_before cells m) vs
  | _ => true
  end.

Lemma older_no_mark (frames : list frame) (cells : list (location * mark))
    (vs : list value) (w : value) :
  value_no_mark frames w = true -> older frames cells vs w = true.
Proof.
  intros H. destruct w; try reflexivity. cbn [value_no_mark older] in H |- *.
  rewrite H. reflexivity.
Qed.

Lemma older_incl (frames : list frame) (cells : list (location * mark))
    (vs vs' : list value) (w : value) :
  (∀ v : value, In v vs' -> In v vs) ->
  older frames cells vs w = true ->
  older frames cells vs' w = true.
Proof.
  intros Hincl. destruct w; try (intros _; reflexivity). cbn [older].
  intros H.
  apply orb_true_iff in H as [H | H]; [ rewrite H; reflexivity | ].
  apply orb_true_iff. right. apply forallb_forall. intros v Hin.
  exact (forallb_in _ _ v H (Hincl v Hin)).
Qed.

Lemma older_frames (frames frames' : list frame)
    (cells : list (location * mark)) (next : symbol) (vs : list value)
    (w : value) :
  (∀ s : symbol,
    (s < next)%positive ->
    on_stack s frames = false ->
    on_stack s frames' = false) ->
  value_below next w = true ->
  older frames cells vs w = true ->
  older frames' cells vs w = true.
Proof.
  intros Hnew. destruct w; try (intros _ _; reflexivity).
  cbn [older value_below].
  intros Hb H. unfold symbol_below in Hb. apply Pos.ltb_lt in Hb.
  destruct (on_stack s frames) eqn:Hon;
    [ simpl in H; rewrite H; apply orb_true_r | ].
  rewrite (Hnew s Hb Hon). reflexivity.
Qed.

Lemma cont_held_older_frames (frames frames' : list frame)
    (cells : list (location * mark)) (base next : symbol) (c : cont_value) :
  (∀ s : symbol,
    (s < next)%positive ->
    on_stack s frames = false ->
    on_stack s frames' = false) ->
  cont_below base next c = true ->
  cont_held_older frames cells c = true ->
  cont_held_older frames' cells c = true.
Proof.
  intros Hnew.
  induction c
    as [h vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_deep_ind;
    simpl.
  - intros Hb Ho. apply andb_true_iff in Hb as [Hb _].
    apply andb_true_iff in Hb as [Hb Hbc].
    apply andb_true_iff in Hb as [_ Hbv].
    apply andb_true_iff in Ho as [Hov Hoc]. apply andb_true_iff. split.
    + unfold held_older in Hov |- *. apply forallb_forall. intros [x w] Hin.
      exact
        (older_frames _ _ _ next _ w Hnew (forallb_in _ _ (x, w) Hbv Hin)
           (forallb_in _ _ (x, w) Hov Hin)).
    + apply forallb_forall. intros [k c] Hin. rewrite Forall_forall in IH.
      exact
        (IH (k, c) Hin (forallb_in _ _ (k, c) Hbc Hin)
           (forallb_in _ _ (k, c) Hoc Hin)).
  - reflexivity.
  - intros Hb Ho. apply andb_true_iff in Hb as [Hbc _]. exact (IH Hbc Ho).
  - intros Hb Ho. apply andb_true_iff in Hb as [Hb Hbe].
    apply andb_true_iff in Hb as [_ Hbt].
    apply andb_true_iff in Ho as [Hot Hoe].
    rewrite (IHreturn Hbt Hot), (IHexn Hbe Hoe). reflexivity.
Qed.

(* What a continuation has held is among what the continuations of its
   environment have held. *)
Lemma resolve_held (k : continuation) (conts : cont_env) (v : value) :
  In v (cont_held (resolve k conts)) -> In v (conts_held conts).
Proof.
  unfold resolve. destruct (lookup k conts) as [c | ] eqn:Hlookup;
    [ | intros [] ].
  intros Hv. unfold conts_held. apply in_flat_map. exists (k, c).
  exact (conj (lookup_in _ _ _ Hlookup) Hv).
Qed.

Lemma rec_closures_held
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) (v : value) :
  In v (conts_held (rec_closures handlers vars conts)) ->
  In v (conts_held conts).
Proof.
  unfold conts_held, rec_closures. generalize handlers at 1 as all.
  intros all.
  induction handlers as [ | [k h] handlers IH]; simpl; [ intros [] | ].
  intros Hin. apply in_app_iff in Hin as [Hin | Hin];
    [ exact Hin | exact (IH Hin) ].
Qed.

Lemma rec_closures_older (frames : list frame)
    (cells : list (location * mark))
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  held_older frames cells vars (conts_held conts) = true ->
  forallb (fun '(_, c) => cont_held_older frames cells c) conts = true ->
  forallb (fun '(_, c) => cont_held_older frames cells c)
    (rec_closures handlers vars conts)
  = true.
Proof.
  intros Hv Hc. unfold rec_closures. generalize handlers at 1 as all.
  intros all.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  apply andb_true_iff. split; [ | exact IH ].
  apply andb_true_iff. split; [ exact Hv | exact Hc ].
Qed.

Lemma bind_params_older (frames : list frame)
    (cells : list (location * mark)) (held : list value)
    (params : list variable) (vs : list value) (vars vars' : env) :
  Forall (fun v => older frames cells held v = true) vs ->
  held_older frames cells vars held = true ->
  bind_params params vs vars = Some vars' ->
  held_older frames cells vars' held = true.
Proof.
  intros Hvs Hvars Hbind.
  apply (forallb_pairs_Forall (older frames cells held)).
  apply
    (bind_params_values_all (fun v => older frames cells held v = true)
       params vs vars vars' Hvs);
    [ | exact Hbind ].
  apply (forallb_pairs_Forall (older frames cells held)). exact Hvars.
Qed.

(* What a continuation has held names no symbol at or above the
   counter. *)
Lemma cont_held_below (base next : symbol) (c : cont_value) :
  cont_below base next c = true ->
  forallb (value_below next) (cont_held c) = true.
Proof.
  induction c
    as [h vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_deep_ind;
    simpl.
  - intros Hb. apply andb_true_iff in Hb as [Hb _].
    apply andb_true_iff in Hb as [_ Hbc].
    rewrite forallb_flat_map. apply forallb_forall. intros [k c] Hin.
    rewrite Forall_forall in IH.
    exact (IH (k, c) Hin (forallb_in _ _ (k, c) Hbc Hin)).
  - reflexivity.
  - intros Hb. apply andb_true_iff in Hb as [Hbc Hbe].
    rewrite forallb_app, Hbe, (IH Hbc). reflexivity.
  - intros Hb. apply andb_true_iff in Hb as [Hb Hbe].
    apply andb_true_iff in Hb as [_ Hbt].
    rewrite forallb_app, (IHreturn Hbt), (IHexn Hbe). reflexivity.
Qed.

(* The value of a simple of the term is older than what is held, as the
   values of the variables are. *)
Lemma eval_simple_older (rs : region_state) (held : list value)
    (sim : simple) (w : value) :
  symbols_fresh rs = true ->
  held_older (rs_frames rs) (rs_cells rs) (state_vars (rs_state rs)) held
  = true ->
  simple_below (rs_base rs) sim = true ->
  eval_simple (state_vars (rs_state rs)) sim = Some w ->
  older (rs_frames rs) (rs_cells rs) held w = true.
Proof.
  intros Hfresh Hvars Hbelow. destruct sim as [x | sym | c]; simpl.
  - exact
      (lookup_forallb (older (rs_frames rs) (rs_cells rs) held) x w _ Hvars).
  - intros [= <-]. cbn [older].
    rewrite (not_drawn_not_on_stack rs sym Hfresh); [ reflexivity | ].
    left. apply Pos.ltb_lt. exact Hbelow.
  - intros [= <-]. reflexivity.
Qed.

(* The tests under another list of cells. [same_before] is what is
   asked of each value that is held: the test of it against a mark
   passes under the other list if it passes under this one. *)
Definition same_before (cells cells' : list (location * mark)) (v : value)
    : Prop :=
  ∀ m : mark, held_before cells m v = true -> held_before cells' m v = true.

Lemma conts_values_all_in (P : value -> Prop) (conts : cont_env)
    (k : continuation) (c : cont_value) :
  conts_values_all P conts -> In (k, c) conts -> cont_values_all P c.
Proof.
  unfold conts_values_all.
  induction conts as [ | [k' c'] conts IH]; cbn [fold_right snd];
    [ intros _ [] | ].
  intros [Hc Hrest] [[= <- <-] | Hin]; [ exact Hc | exact (IH Hrest Hin) ].
Qed.

(* What a continuation has held is among the values that it holds. *)
Lemma cont_held_all (P : value -> Prop) (c : cont_value) :
  cont_values_all P c -> Forall P (cont_held c).
Proof.
  induction c
    as [h vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_deep_ind;
    cbn [cont_values_all cont_held].
  - intros [_ Hconts]. apply Forall_forall. intros v Hin.
    apply in_flat_map in Hin as ([k c] & Hc & Hin).
    rewrite Forall_forall in IH.
    exact
      (proj1 (Forall_forall _ _)
         (IH (k, c) Hc (conts_values_all_in P conts k c Hconts Hc)) v Hin).
  - intros _. apply Forall_nil.
  - intros [Hextra Hc]. apply Forall_app.
    split; [ exact Hextra | exact (IH Hc) ].
  - intros (_ & Hr & He). apply Forall_app.
    split; [ exact (IHreturn Hr) | exact (IHexn He) ].
Qed.

Lemma conts_held_all (P : value -> Prop) (conts : cont_env) :
  conts_values_all P conts -> Forall P (conts_held conts).
Proof.
  intros Hconts. apply Forall_forall. intros v Hin. unfold conts_held in Hin.
  apply in_flat_map in Hin as ([k c] & Hc & Hin).
  exact
    (proj1 (Forall_forall _ _)
       (cont_held_all P c (conts_values_all_in P conts k c Hconts Hc)) v Hin).
Qed.

Lemma older_cells (frames : list frame) (cells cells' : list (location * mark))
    (vs : list value) (w : value) :
  Forall (same_before cells cells') vs ->
  older frames cells vs w = true ->
  older frames cells' vs w = true.
Proof.
  intros Hvs. destruct w; try (intros _; reflexivity). cbn [older].
  intros H.
  apply orb_true_iff in H as [H | H]; [ rewrite H; reflexivity | ].
  apply orb_true_iff. right. apply forallb_forall. intros v Hin.
  exact (proj1 (Forall_forall _ vs) Hvs v Hin s (forallb_in _ _ v H Hin)).
Qed.

Lemma cont_held_older_cells (frames : list frame)
    (cells cells' : list (location * mark)) (c : cont_value) :
  cont_values_all (same_before cells cells') c ->
  cont_held_older frames cells c = true ->
  cont_held_older frames cells' c = true.
Proof.
  induction c
    as [h vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_deep_ind;
    cbn [cont_values_all cont_held_older].
  - intros [_ Hconts] Ho. apply andb_true_iff in Ho as [Hov Hoc].
    apply andb_true_iff. split.
    + unfold held_older in Hov |- *. revert Hov. apply forallb_impl.
      intros [x w] _. apply older_cells.
      exact (conts_held_all _ conts Hconts).
    + apply forallb_forall. intros [k c] Hin. rewrite Forall_forall in IH.
      exact
        (IH (k, c) Hin (conts_values_all_in _ conts k c Hconts Hin)
           (forallb_in _ _ (k, c) Hoc Hin)).
  - reflexivity.
  - intros [_ Hc]. exact (IH Hc).
  - intros (_ & Hr & He) Ho. apply andb_true_iff in Ho as [Hor Hoe].
    rewrite (IHreturn Hr Hor), (IHexn He Hoe). reflexivity.
Qed.

(* A value all of whose locations the heap binds is not a location
   that the heap does not bind: to list that location changes nothing
   of what the test says of the value. *)
Lemma same_before_unbound (hp : heap) (cells : list (location * mark))
    (l : location) (m' : mark) (v : value) :
  is_bound l hp = false ->
  value_closed hp v ->
  same_before cells ((l, m') :: cells) v.
Proof.
  intros Hl Hclosed m.
  destruct v as [ | | | s | | | | l0 | ]; try (intros H; exact H).
  cbn [held_before]. rewrite lookup_cons.
  destruct (Pos.eqb_spec l0 l) as [-> | _]; [ | intros H; exact H ].
  rewrite (Hclosed l (or_introl eq_refl)) in Hl. discriminate Hl.
Qed.

(* The tests of the third and fourth parts of [calls_kept], and of the
   fifth, of the variables and of the continuations of a state. *)
Definition calls_held (rs : region_state) (vars : env) (conts : cont_env)
    : Prop :=
  forallb (value_usable rs) (conts_held conts) = true
  ∧ held_older (rs_frames rs) (rs_cells rs) vars (conts_held conts) = true
  ∧ forallb (fun '(_, c) => cont_held_older (rs_frames rs) (rs_cells rs) c)
      conts
    = true.

Lemma calls_kept_parts (rs : region_state) :
  calls_kept rs = true
  <-> marks_in_variables rs = true
      ∧ frames_ordered rs = true
      ∧ calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)).
Proof.
  unfold calls_kept, calls_held, held. cbv zeta. rewrite !andb_true_iff.
  tauto.
Qed.

(* Continuations that have held no more than those of [conts]. *)
Lemma calls_held_closures (rs : region_state) (vars : env)
    (conts conts' : cont_env) :
  (∀ v : value, In v (conts_held conts') -> In v (conts_held conts)) ->
  forallb (fun '(_, c) => cont_held_older (rs_frames rs) (rs_cells rs) c)
    conts'
  = true ->
  forallb (value_usable rs) (conts_held conts) = true ->
  held_older (rs_frames rs) (rs_cells rs) vars (conts_held conts) = true ->
  calls_held rs vars conts'.
Proof.
  intros Hincl Holder Hu Hvars. split; [ | split; [ | exact Holder ] ].
  - apply forallb_forall. intros v Hin.
    exact (forallb_in _ _ v Hu (Hincl v Hin)).
  - unfold held_older in Hvars |- *. revert Hvars. apply forallb_impl.
    intros [x w] _. exact (older_incl _ _ _ _ w Hincl).
Qed.

(* What is asked of a continuation, and of a value that a continuation
   is given. *)
Definition cont_calls (rs : region_state) (c : cont_value) : Prop :=
  cont_no_mark (rs_frames rs) c = true
  ∧ forallb (value_usable rs) (cont_held c) = true
  ∧ cont_held_older (rs_frames rs) (rs_cells rs) c = true.

Definition given (rs : region_state) (v : value) : Prop :=
  no_mark (rs_frames rs) v ∧ value_usable rs v = true.

Lemma calls_extra (rs : region_state) (c : cont_value) (extra : list value) :
  cont_calls rs (Extra_args c extra)
  <-> cont_calls rs c ∧ Forall (given rs) extra.
Proof.
  unfold cont_calls, given, no_mark. simpl.
  rewrite forallb_app, !andb_true_iff, !forallb_Forall. split.
  - intros ((Hc & He) & (Hue & Huc) & Ho).
    split; [ exact (conj Hc (conj Huc Ho)) | ].
    rewrite Forall_forall in *. intros v Hin.
    exact (conj (He v Hin) (Hue v Hin)).
  - intros ((Hc & Huc & Ho) & Hg).
    refine (conj (conj Hc _) (conj (conj _ Huc) Ho)).
    + exact (Forall_impl _ (fun v H => proj1 H) Hg).
    + exact (Forall_impl _ (fun v H => proj2 H) Hg).
Qed.

Lemma calls_over (rs : region_state) (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_calls rs
    (Over_application remaining remaining_arity return_arity return_to exn_to)
  <-> Forall (no_mark (rs_frames rs)) remaining
      ∧ cont_calls rs return_to
      ∧ cont_calls rs exn_to.
Proof.
  unfold cont_calls. simpl.
  rewrite forallb_app, !andb_true_iff, forallb_Forall. tauto.
Qed.

Lemma calls_code (rs : region_state) (closure : value) (c : code)
    (vs : list value) (callee_vars : env) (return_to exn_to : cont_value) :
  no_mark (rs_frames rs) closure ->
  Forall (no_mark (rs_frames rs)) vs ->
  cont_calls rs return_to ->
  cont_calls rs exn_to ->
  bind_params (code_params c) vs [(code_my_closure c, closure)]
  = Some callee_vars ->
  calls_held rs callee_vars
    [(code_return_continuation c, return_to);
     (code_exn_continuation c, exn_to)].
Proof.
  intros Hclosure Hvs (_ & Hur & Hor) (_ & Hue & Hoe) Hbind.
  split; [ | split ].
  - simpl. rewrite app_nil_r, forallb_app, Hur, Hue. reflexivity.
  - apply (forallb_pairs_Forall (older (rs_frames rs) (rs_cells rs) _)).
    refine
      (Forall_impl _ _
         (bind_params_values_all (no_mark (rs_frames rs)) _ _ _ _ Hvs
            (Forall_cons (code_my_closure c, closure) Hclosure
               (Forall_nil _))
            Hbind)).
    intros [x w]. exact (older_no_mark _ _ _ w).
  - simpl. rewrite Hor, Hoe. reflexivity.
Qed.

Lemma calls_handler (rs : region_state) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (def_vars : env)
    (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (handler_vars : env) :
  cont_calls rs
    (Closure (Handler params metadata handler) def_vars def_conts recursive) ->
  Forall
    (fun v =>
       older (rs_frames rs) (rs_cells rs) (conts_held def_conts) v = true)
    vs ->
  bind_params params vs def_vars = Some handler_vars ->
  calls_held rs handler_vars
    (rec_closures recursive def_vars def_conts ++ def_conts).
Proof.
  intros (_ & Hu & Ho) Hvs Hbind. simpl in Ho.
  apply andb_true_iff in Ho as [Hvars Hconts].
  apply (calls_held_closures rs handler_vars def_conts).
  - intros v Hin. unfold conts_held in Hin. rewrite flat_map_app in Hin.
    apply in_app_iff in Hin as [Hin | Hin];
      [ exact (rec_closures_held _ _ _ v Hin) | exact Hin ].
  - rewrite forallb_app,
      (rec_closures_older _ _ recursive def_vars def_conts Hvars Hconts),
      Hconts.
    reflexivity.
  - exact Hu.
  - exact (bind_params_older _ _ _ params vs def_vars handler_vars Hvs Hvars
             Hbind).
Qed.

Lemma resolve_calls (rs : region_state) (k : continuation)
    (conts : cont_env) :
  forallb (fun '(_, c) => cont_no_mark (rs_frames rs) c) conts = true ->
  forallb (value_usable rs) (conts_held conts) = true ->
  forallb (fun '(_, c) => cont_held_older (rs_frames rs) (rs_cells rs) c)
    conts
  = true ->
  cont_calls rs (resolve k conts).
Proof.
  intros Hm Hu Ho. split; [ exact (resolve_no_mark _ k conts Hm) | ]. split.
  - apply forallb_forall. intros v Hin.
    exact (forallb_in _ _ v Hu (resolve_held k conts v Hin)).
  - unfold resolve.
    destruct (lookup k conts) as [c | ] eqn:Hlookup; [ | reflexivity ].
    exact
      (lookup_forallb (cont_held_older (rs_frames rs) (rs_cells rs)) k c conts
         Ho Hlookup).
Qed.

(* ======================================================================== *)
(* A step keeps the other parts of the test                                 *)
(* ======================================================================== *)

(* What is shown of a step, of a state that has the first part. *)
Definition calls_step (result : region_step_result) : Prop :=
  match result with
  | Region_step rs' =>
    calls_held rs' (state_vars (rs_state rs')) (state_conts (rs_state rs'))
  | Region_exit _ vs rs' => forallb (value_usable rs') vs = true
  | Region_undefined _ => True
  end.

Lemma plain_step_calls (rs : region_state) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  forallb (simple_usable rs) (evaluated_simples (state_expr (rs_state rs)))
  = true ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
     (v : value) (hp' : heap) (rd' : read_record),
    state_expr (rs_state rs) = Let x kind d body ->
    exec_named (live_statics rs) (state_heap (rs_state rs))
      (state_reads (rs_state rs)) (state_vars (rs_state rs)) d
    = Some (v, hp', rd') ->
    value_no_mark (rs_frames rs) v = true
    ∧ heap_no_mark (rs_frames rs) hp') ->
  calls_step (plain_step rs).
Proof.
  intros Hfresh Hmarks (C1 & C2 & C3) Husable Hlet.
  pose proof Hmarks as HM.
  apply marks_in_variables_parts in HM as (M1 & M2 & _ & _).
  assert (Hstep :
    call_result (given rs) (calls_held rs)
      (step (with_statics (rs_state rs) (live_statics rs)))).
  { apply
      (step_call_result (no_mark (rs_frames rs)) (given rs) (cont_calls rs)
         (calls_held rs) (live_statics rs));
      cbn [with_statics state_expr state_vars state_conts state_statics
           state_heap state_reads].
    - exact (no_mark_partial _).
    - exact (no_mark_fields rs Hmarks).
    - intros v [H _]. exact H.
    - intros callee args args_arity H. exact (conj H eq_refl).
    - exact (calls_extra rs).
    - exact (calls_over rs).
    - exact (calls_code rs).
    - intros params metadata handler def_vars def_conts recursive vs
        handler_vars Hc Hvs.
      apply (calls_handler rs _ _ _ _ _ _ vs handler_vars Hc).
      exact (Forall_impl _ (fun v H => older_no_mark _ _ _ v (proj1 H)) Hvs).
    - reflexivity.
    - exact (conj C1 (conj C2 C3)).
    - intros k. exact (resolve_calls rs k _ M2 C1 C3).
    - intros x kind d body v hp' rd' He Hd. split; [ exact C1 | ].
      split; [ | exact C3 ].
      apply andb_true_iff. split; [ | exact C2 ].
      exact
        (older_no_mark _ _ _ v (proj1 (Hlet x kind d body v hp' rd' He Hd))).
    - intros k h body _.
      apply (calls_held_closures rs _ (state_conts (rs_state rs)));
        [ | | exact C1 | exact C2 ].
      + intros v Hin. apply in_app_iff in Hin as [Hin | Hin]; exact Hin.
      + apply andb_true_iff. split; [ | exact C3 ].
        apply andb_true_iff. split; [ exact C2 | exact C3 ].
    - intros handlers body _.
      apply (calls_held_closures rs _ (state_conts (rs_state rs)));
        [ | | exact C1 | exact C2 ].
      + intros v Hin. unfold conts_held in Hin. rewrite flat_map_app in Hin.
        apply in_app_iff in Hin as [Hin | Hin];
          [ exact (rec_closures_held _ _ _ v Hin) | exact Hin ].
      + rewrite forallb_app, (rec_closures_older _ _ handlers _ _ C2 C3), C3.
        reflexivity.
    - intros sim v Hin Hv. split.
      + exact
          (simple_no_mark rs _ sim v Hfresh (fewer_marks_same rs) M1
             (forallb_in _ _ sim Husable Hin) Hv).
      + exact
          (simple_usable_value rs sim v (forallb_in _ _ sim Husable Hin) Hv). }
  unfold plain_step.
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k vs hp rd | ]; [ exact Hstep | | exact I ].
  apply forallb_forall. intros v Hin.
  exact (proj2 (proj1 (Forall_forall _ vs) Hstep v Hin)).
Qed.

Lemma region_apply_cont_calls (rs : region_state) (ac : apply_cont) :
  symbols_fresh rs = true ->
  marks_in_variables rs = true ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  state_expr (rs_state rs) = Apply_cont ac ->
  ((∀ (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)),
     resolve (ac_continuation ac) (state_conts (rs_state rs))
     <> Closure h vars conts recursive) ->
   forallb (simple_usable rs) (ac_args ac) = true) ->
  calls_step (region_apply_cont rs ac).
Proof.
  intros Hfresh Hmarks Hcalls He Husable. unfold region_apply_cont.
  pose proof Hcalls as (C1 & C2 & C3). pose proof Hmarks as HM.
  apply marks_in_variables_parts in HM as (_ & M2 & _ & _).
  pose proof (resolve_calls rs (ac_continuation ac) _ M2 C1 C3) as Hc.
  pose proof (resolve_held (ac_continuation ac) (state_conts (rs_state rs)))
    as Hheld.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try (apply (plain_step_calls rs Hfresh Hmarks Hcalls);
         [ rewrite He; apply Husable; intros; discriminate
         | intros x kind d body v hp' rd' Hlet; rewrite He in Hlet;
           discriminate ]).
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ]
    eqn:Hargs; [ | exact I ].
  destruct (kinds_eqb _ _); [ | exact I ].
  destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
    [ | exact I ].
  apply
    (calls_handler rs params metadata handler def_vars def_conts recursive vs
       handler_vars Hc);
    [ | exact Hbind ].
  apply (eval_simples_forall _ (state_vars (rs_state rs)) (ac_args ac) vs);
    [ | exact Hargs ].
  intros sim w Hin Hw.
  apply (older_incl _ _ (conts_held (state_conts (rs_state rs))) _ w Hheld).
  apply (eval_simple_older rs _ sim w Hfresh C2); [ | exact Hw ].
  apply (evaluated_simples_below _ (Apply_cont ac));
    [ rewrite <- He; exact (symbols_fresh_expr rs Hfresh) | exact Hin ].
Qed.

(* The tests of the fourth and fifth parts, under other frames. *)
Lemma older_frames_state (rs : region_state) (frames' : list frame) :
  symbols_fresh rs = true ->
  (∀ s : symbol,
    (s < rs_next rs)%positive ->
    on_stack s (rs_frames rs) = false ->
    on_stack s frames' = false) ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  held_older frames' (rs_cells rs) (state_vars (rs_state rs))
    (conts_held (state_conts (rs_state rs)))
  = true
  ∧ forallb (fun '(_, c) => cont_held_older frames' (rs_cells rs) c)
      (state_conts (rs_state rs))
    = true.
Proof.
  intros Hfresh Hnew (_ & C2 & C3).
  destruct (symbols_fresh_held rs Hfresh) as (B1 & B2 & _ & _).
  split; apply forallb_forall.
  - intros [x w] Hin.
    exact
      (older_frames _ _ _ _ _ w Hnew (forallb_in _ _ (x, w) B1 Hin)
         (forallb_in _ _ (x, w) C2 Hin)).
  - intros [k c] Hin.
    exact
      (cont_held_older_frames _ _ _ (rs_base rs) _ c Hnew
         (forallb_in _ _ (k, c) B2 Hin) (forallb_in _ _ (k, c) C3 Hin)).
Qed.

(* A state that binds what the state before binds, and in which no
   cell is dead that was not. *)
Lemma value_usable_more (rs rs' : region_state) (v : value) :
  rs_base rs' = rs_base rs ->
  (∀ s : symbol,
    is_some (lookup s (static_values (live_statics rs))) = true ->
    is_some (lookup s (static_values (live_statics rs'))) = true) ->
  (∀ l : location, cell_dead rs l = false -> cell_dead rs' l = false) ->
  value_usable rs v = true ->
  value_usable rs' v = true.
Proof.
  intros Hbase Hmore Hcells.
  destruct v as [ | | | s | | | | l | ]; try reflexivity.
  - unfold value_usable, drawn. rewrite Hbase. intros H.
    apply orb_true_iff in H as [H | H]; [ rewrite H; reflexivity | ].
    rewrite (Hmore s H). apply orb_true_r.
  - unfold value_usable. rewrite !negb_true_iff. exact (Hcells l).
Qed.

(* No cell is dead that was not, where the cells are listed as they
   were and the stack has the marks that it had. *)
Lemma cell_dead_more (rs rs' : region_state) (l : location) :
  rs_cells rs' = rs_cells rs ->
  (∀ m : mark,
    on_stack m (rs_frames rs) = true -> on_stack m (rs_frames rs') = true) ->
  cell_dead rs l = false ->
  cell_dead rs' l = false.
Proof.
  intros Hcells Hstack. unfold cell_dead. rewrite Hcells.
  destruct (lookup l (rs_cells rs)) as [m | ]; [ | intros _; reflexivity ].
  rewrite !negb_false_iff. exact (Hstack m).
Qed.

Lemma begin_region_calls (rs : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr) :
  symbols_fresh rs = true ->
  (∀ (l : location) (m : mark),
    In (l, m) (rs_cells rs) -> (m < rs_next rs)%positive) ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  calls_step (begin_region rs x kind mark_kind body).
Proof.
  intros Hfresh Hlisted Hcalls. unfold begin_region.
  destruct (kind_eqb kind Region); [ | exact I ].
  destruct
    (older_frames_state rs (Frame (rs_next rs) mark_kind [] :: rs_frames rs)
       Hfresh)
    as [N2 N3]; [ | exact Hcalls | ].
  { intros s Hlt Hoff.
    change
      (Pos.eqb (rs_next rs) s || on_stack s (rs_frames rs) = false).
    rewrite Hoff, orb_false_r. apply Pos.eqb_neq. lia. }
  destruct Hcalls as (C1 & _ & _).
  split; [ | split; [ | exact N3 ] ].
  - revert C1. apply forallb_impl. intros v _.
    apply value_usable_more; [ reflexivity | intros s H; exact H | ].
    intros l. apply cell_dead_more; [ reflexivity | ].
    intros m H.
    change (Pos.eqb (rs_next rs) m || on_stack m (rs_frames rs) = true).
    rewrite H. apply orb_true_r.
  - apply andb_true_iff. split; [ | exact N2 ].
    apply orb_true_iff. right.
    destruct (symbols_fresh_held rs Hfresh) as (_ & B2 & _ & _).
    apply forallb_forall. intros v Hin.
    unfold conts_held in Hin. apply in_flat_map in Hin as ([k c] & Hc & Hin).
    pose proof
      (forallb_in _ _ v
         (cont_held_below _ _ c (forallb_in _ _ (k, c) B2 Hc)) Hin)
      as Hv.
    destruct v as [ | | | s | | | | l | ]; try reflexivity;
      [ exact Hv | ].
    cbn [held_before bind_let rs_cells].
    destruct (lookup l (rs_cells rs)) as [m | ] eqn:Hlookup;
      [ | reflexivity ].
    apply Pos.ltb_lt. exact (Hlisted l m (lookup_in _ _ _ Hlookup)).
Qed.

(* The frame that the end of a region cuts down to is that of its
   mark. *)
Lemma cut_mark (m : mark) (frames above below : list frame) :
  cut m frames = Some (above, below) ->
  ∃ f : frame, frames = above ++ f :: below ∧ frame_mark f = m.
Proof.
  revert above below.
  induction frames as [ | f frames IH]; intros above below; simpl;
    [ discriminate | ].
  destruct (Pos.eqb (frame_mark f) m) eqn:Heq.
  - intros [= <- <-]. exists f.
    split; [ reflexivity | apply Pos.eqb_eq; exact Heq ].
  - destruct (cut m frames) as [[above' below'] | ]; simpl;
      [ | discriminate ].
    intros [= <- <-]. destruct (IH _ _ eq_refl) as (f' & -> & Hf').
    exists f'. split; [ reflexivity | exact Hf' ].
Qed.

(* The mark of each frame is below what is above the frames. *)
Lemma frames_ordered_below_marks (frames : list frame) :
  ∀ (above : symbol) (g : frame),
    frames_ordered_below above frames = true ->
    In g frames ->
    (frame_mark g < above)%positive.
Proof.
  induction frames as [ | f below IH]; intros above g; simpl;
    [ intros _ [] | ].
  rewrite !andb_true_iff. intros [[Hm _] Hb] [<- | Hin];
    [ apply Pos.ltb_lt; exact Hm | ].
  apply Pos.ltb_lt in Hm. specialize (IH _ g Hb Hin). lia.
Qed.

(* What the frames down to and with [f] bind was drawn after the mark
   of [f]. *)
Lemma frames_ordered_cut (above0 : list frame) :
  ∀ (above : symbol) (f : frame) (below : list frame) (s : symbol)
    (allocated : statically_allocated_value),
    frames_ordered_below above (above0 ++ f :: below) = true ->
    In (s, allocated) (live_locals (above0 ++ [f])) ->
    (frame_mark f < s)%positive.
Proof.
  unfold live_locals.
  induction above0 as [ | g above0 IH]; intros above f below s allocated;
    simpl; rewrite !andb_true_iff; intros [[_ Hl] Hb] Hin.
  - rewrite app_nil_r in Hin.
    pose proof (forallb_in _ _ (s, allocated) Hl Hin) as H. simpl in H.
    apply andb_true_iff in H as [H _]. apply Pos.ltb_lt. exact H.
  - apply in_app_iff in Hin as [Hin | Hin];
      [ | exact (IH _ f below s allocated Hb Hin) ].
    pose proof (forallb_in _ _ (s, allocated) Hl Hin) as H. simpl in H.
    apply andb_true_iff in H as [H _]. apply Pos.ltb_lt in H.
    pose proof
      (frames_ordered_below_marks _ _ f Hb
         (in_or_app above0 (f :: below) f (or_intror (or_introl eq_refl))))
      as Hf.
    lia.
Qed.

(* The marks of the frames above a frame are above its mark. *)
Lemma frames_ordered_above (above0 : list frame) :
  ∀ (above : symbol) (f : frame) (below : list frame) (g : frame),
    frames_ordered_below above (above0 ++ f :: below) = true ->
    In g above0 ->
    (frame_mark f < frame_mark g)%positive.
Proof.
  induction above0 as [ | g0 above0 IH]; intros above f below g;
    cbn [app frames_ordered_below]; [ intros _ [] | ].
  rewrite !andb_true_iff. intros [_ Hb] [<- | Hin];
    [ | exact (IH _ f below g Hb Hin) ].
  exact
    (frames_ordered_below_marks _ _ f Hb
       (in_or_app above0 (f :: below) f (or_intror (or_introl eq_refl)))).
Qed.

(* A mark that is on the stack and is smaller than the mark of a frame
   is the mark of a frame below it. *)
Lemma on_stack_below (above0 : list frame) (above : symbol) (f : frame)
    (below : list frame) (m : mark) :
  frames_ordered_below above (above0 ++ f :: below) = true ->
  on_stack m (above0 ++ f :: below) = true ->
  (m < frame_mark f)%positive ->
  on_stack m below = true.
Proof.
  intros Hord Hon Hlt. rewrite on_stack_app in Hon.
  apply orb_true_iff in Hon as [Hon | Hon].
  - unfold on_stack in Hon. apply existsb_exists in Hon as (g & Hg & Heq).
    apply Pos.eqb_eq in Heq.
    pose proof (frames_ordered_above above0 above f below g Hord Hg). lia.
  - change (on_stack m (f :: below))
      with (Pos.eqb (frame_mark f) m || on_stack m below)
      in Hon.
    apply orb_true_iff in Hon as [Heq | Hon]; [ | exact Hon ].
    apply Pos.eqb_eq in Heq. lia.
Qed.

(* What passes [held_before] against the mark of the region that ends
   may be evaluated after as before. A symbol drawn before that mark:
   the frames that are popped bind none. A location listed with a
   smaller mark: the frame of that mark is below the frames that are
   popped. *)
Lemma end_region_usable (rs rs' : region_state) (above : list frame)
    (f : frame) (below : list frame) (v : value) :
  rs_base rs' = rs_base rs ->
  state_statics (rs_state rs') = state_statics (rs_state rs) ->
  rs_cells rs' = rs_cells rs ->
  rs_frames rs = above ++ f :: below ->
  rs_frames rs' = below ->
  frames_ordered rs = true ->
  held_before (rs_cells rs) (frame_mark f) v = true ->
  value_usable rs v = true ->
  value_usable rs' v = true.
Proof.
  intros Hbase Hstatics Hcells Hframes Hframes' Hord.
  destruct v as [ | | | s | | | | l | ]; try (intros _ _; reflexivity);
    cbn [held_before].
  - intros Hold. apply Pos.ltb_lt in Hold. unfold value_usable, drawn.
    rewrite Hbase, !live_statics_values, Hstatics, Hframes, Hframes'.
    intros H. apply orb_true_iff in H as [H | H]; [ rewrite H; reflexivity | ].
    apply orb_true_iff. right.
    replace (above ++ f :: below) with ((above ++ [f]) ++ below) in H
      by (rewrite <- app_assoc; reflexivity).
    rewrite live_locals_app, <- app_assoc, lookup_app in H.
    destruct (lookup s (live_locals (above ++ [f]))) as [allocated | ]
      eqn:Hlookup; [ | exact H ].
    apply lookup_in in Hlookup. unfold frames_ordered in Hord.
    rewrite Hframes in Hord.
    pose proof (frames_ordered_cut above _ f below s allocated Hord Hlookup).
    lia.
  - unfold value_usable, cell_dead. rewrite Hcells, Hframes'.
    destruct (lookup l (rs_cells rs)) as [m | ]; [ | intros _ _; reflexivity ].
    intros Hold Hon. apply Pos.ltb_lt in Hold.
    rewrite negb_involutive in Hon |- *.
    unfold frames_ordered in Hord. rewrite Hframes in Hord, Hon.
    exact (on_stack_below above _ f below m Hord Hon Hold).
Qed.

Lemma end_region_calls (rs : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  symbols_fresh rs = true ->
  frames_ordered rs = true ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  simple_below (rs_base rs) region = true ->
  calls_step (end_region rs x kind ended_by region body).
Proof.
  intros Hfresh Hord Hcalls Hbelow. unfold end_region. cbv zeta.
  destruct (kind_eqb kind Value); [ | exact I ].
  destruct (eval_simple (state_vars (rs_state rs)) region) as [v | ]
    eqn:Hregion; [ | exact I ].
  destruct v as [ | | | m | | | | | ]; try exact I.
  destruct (cut m (rs_frames rs)) as [[above below] | ] eqn:Hcut;
    [ | exact I ].
  destruct (cut_mark _ _ _ _ Hcut) as (f & Hframes & <-).
  destruct (older_frames_state rs below Hfresh) as [N2 N3];
    [ | exact Hcalls | ].
  { apply fewer_marks_no_new. intros s Hon. rewrite Hframes, on_stack_app.
    change (on_stack s (f :: below))
      with (Pos.eqb (frame_mark f) s || on_stack s below).
    rewrite Hon, !orb_true_r. reflexivity. }
  destruct Hcalls as (C1 & C2 & _).
  assert (Hold :
    forallb (held_before (rs_cells rs) (frame_mark f))
      (conts_held (state_conts (rs_state rs)))
    = true).
  { pose proof
      (eval_simple_older rs _ region _ Hfresh C2 Hbelow Hregion) as H.
    cbn [older] in H. rewrite Hframes, on_stack_app in H.
    change (on_stack (frame_mark f) (f :: below))
      with
        (Pos.eqb (frame_mark f) (frame_mark f)
         || on_stack (frame_mark f) below)
      in H.
    rewrite Pos.eqb_refl, orb_true_r in H. exact H. }
  split; [ | split; [ exact N2 | exact N3 ] ].
  apply forallb_forall. intros v Hin.
  apply (end_region_usable rs _ above f below v); try reflexivity;
    [ exact Hframes | exact Hord | exact (forallb_in _ _ v Hold Hin)
    | exact (forallb_in _ _ v C1 Hin) ].
Qed.

Lemma local_allocation_calls (rs : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  symbols_fresh rs = true ->
  is_cell_allocation d && kind_eqb kind Value = false ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  calls_step (local_allocation rs x y kind d region body).
Proof.
  intros Hfresh Hcell Hcalls.
  destruct (local_allocation rs x y kind d region body)
    as [rs' | k vs rs' | fault] eqn:Hstep;
    [ | contradiction (local_allocation_no_exit _ _ _ _ _ _ _ _ _ _ Hstep)
    | exact I ].
  destruct (local_allocation_step rs rs' x y kind d region body Hcell Hstep)
    as (v & sym & bindings & _ & _ & Hlocalise & _ & Hvars & Hconts & Hbase
        & _ & _ & Hframes).
  destruct (local_allocation_memory rs rs' x y kind d region body Hcell Hstep)
    as (_ & _ & Hcells).
  destruct (localise_makes _ _ _ _ Hlocalise) as (_ & [allocated Hin] & Hbound).
  destruct (Hbound sym allocated Hin) as (Hfrom & _).
  assert (Hfewer : fewer_marks rs (rs_frames rs')).
  { unfold fewer_marks.
    destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
      destruct Hframes as [-> _]; intros s H; exact H. }
  destruct
    (older_frames_state rs (rs_frames rs') Hfresh
       (fewer_marks_no_new rs _ Hfewer) Hcalls)
    as [N2 N3].
  destruct Hcalls as (C1 & _ & _).
  unfold calls_step, calls_held. rewrite Hvars, Hconts, Hcells.
  split; [ | split; [ | exact N3 ] ].
  - revert C1. apply forallb_impl. intros w _.
    apply value_usable_more; [ exact Hbase | | ].
    + intros s. rewrite !live_statics_values.
      destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
        destruct Hframes as [-> Hrest]; rewrite Hrest.
      * simpl. rewrite is_some_lookup_app. intros ->. apply orb_true_r.
      * change
          (live_locals (Frame m mark_kind (bindings ++ locals) :: frames))
          with ((bindings ++ locals) ++ live_locals frames).
        change (live_locals (Frame m mark_kind locals :: frames))
          with (locals ++ live_locals frames).
        rewrite <- !app_assoc, (is_some_lookup_app s bindings). intros ->.
        apply orb_true_r.
    + intros l. apply cell_dead_more; [ exact Hcells | ].
      destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
        destruct Hframes as [-> _]; intros m0 H; exact H.
  - assert (Hsym :
      older (rs_frames rs') (rs_cells rs)
        (conts_held (state_conts (rs_state rs))) (Value_symbol sym)
      = true).
    { cbn [older]. destruct (on_stack sym (rs_frames rs')) eqn:Hon;
        [ | reflexivity ].
      pose proof (Hfewer sym Hon) as H.
      rewrite (not_drawn_not_on_stack rs sym Hfresh (or_intror Hfrom)) in H.
      discriminate H. }
    apply andb_true_iff. split; [ exact Hsym | ].
    apply andb_true_iff. split; [ exact Hsym | exact N2 ].
Qed.

(* A local allocation of a cell. Its location is one that the heap did
   not bind, and the heap binds every location that the state holds
   (FlambdaRegionsInvariantSpec.locations_all_bound): so it is none of
   the values that the continuations have held, and to list it changes
   nothing of what the test says of them. The frames are as they were,
   and the cell that is listed is in the newest. *)
Lemma local_cell_allocation_calls (rs : region_state) (x y : variable)
    (d : named) (region : option simple) (body : expr) :
  bound rs ->
  is_cell_allocation d = true ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  calls_step (local_cell_allocation rs x y d region body).
Proof.
  intros Hb Hcell (C1 & C2 & C3).
  destruct (local_cell_allocation rs x y d region body)
    as [rs' | k vs rs' | fault] eqn:Hstep;
    [ | contradiction (local_cell_allocation_no_exit _ _ _ _ _ _ _ _ _ Hstep)
    | exact I ].
  destruct (local_cell_allocation_step rs rs' x y d region body Hstep)
    as (l & hp & rd & _ & Hexec & Hstate & Hframes & Hbase & _ & Hcells).
  assert (Hl : is_bound l (state_heap (rs_state rs)) = false).
  { destruct (exec_named_heap_cases _ _ _ _ _ _ d _ Hexec)
      as [[Hno _]
         | [(_ & vs & c & _ & _ & Hv & _)
           | (target & new_simple & l0 & new & c & c' & i & _ & _ & _ & _ & _
              & Hv & _)]].
    - rewrite (cell_allocation_changes_heap d Hcell) in Hno. discriminate Hno.
    - injection Hv as ->. unfold is_bound.
      rewrite FlambdaHeapProofs.fresh_location. reflexivity.
    - discriminate Hv. }
  apply bound_parts in Hb. cbv zeta in Hb. destruct Hb as ([_ Hclosed] & _).
  assert (Hlisted :
    ∃ cells' : list (location * mark),
      rs_cells rs' = cells'
      ∧ conts_values_all (same_before (rs_cells rs) cells')
          (state_conts (rs_state rs))).
  { rewrite Hcells.
    destruct (rs_frames rs) as [ | f frames]; eexists;
      (split; [ reflexivity | ]); revert Hclosed;
      apply conts_values_all_impl; intros v Hv.
    - intros m H. exact H.
    - exact (same_before_unbound _ _ l (frame_mark f) v Hl Hv). }
  destruct Hlisted as (cells' & Hcells' & Hsame).
  unfold calls_step, calls_held. rewrite Hcells', Hframes, Hstate.
  cbn [state_vars state_conts].
  split; [ | split ].
  - revert C1. apply forallb_impl. intros v _.
    apply value_usable_more; [ exact Hbase | | ].
    + intros s. rewrite !live_statics_values, Hframes, Hstate.
      intros H. exact H.
    + intros l0. unfold cell_dead. rewrite Hframes, Hcells.
      destruct (rs_frames rs) as [ | f frames]; [ intros H; exact H | ].
      rewrite lookup_cons.
      destruct (Pos.eqb l0 l); [ | intros H; exact H ].
      intros _. apply negb_false_iff.
      change
        (Pos.eqb (frame_mark f) (frame_mark f)
         || on_stack (frame_mark f) frames
         = true).
      rewrite Pos.eqb_refl. reflexivity.
  - apply andb_true_iff. split; [ reflexivity | ].
    apply andb_true_iff. split; [ reflexivity | ].
    unfold held_older in C2 |- *. revert C2. apply forallb_impl.
    intros [x0 w] _. apply older_cells.
    exact (conts_held_all _ _ Hsame).
  - apply forallb_forall. intros [k c] Hin.
    exact
      (cont_held_older_cells _ _ _ c (conts_values_all_in _ _ k c Hsame Hin)
         (forallb_in _ _ (k, c) C3 Hin)).
Qed.

Lemma static_step_calls (rs : region_state) (bindings : list static_binding)
    (body : expr) :
  state_expr (rs_state rs) = Let_static bindings body ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  calls_step (static_step rs).
Proof.
  intros He (C1 & C2 & C3).
  destruct (static_step rs) as [rs' | k vs rs' | fault] eqn:Hstep;
    [ | contradiction (static_step_no_exit rs bindings body _ _ _ He Hstep)
    | exact I ].
  destruct (static_step_step rs rs' bindings body He Hstep)
    as (statics' & Heval & ->).
  destruct (eval_static_bindings_parts _ _ _ _ Heval)
    as (new_code & new_values & _ & Hvalues & _ & _).
  split; [ | exact (conj C2 C3) ].
  revert C1. apply forallb_impl. intros w _.
  apply value_usable_more; [ reflexivity | | intros l H; exact H ].
  intros s. rewrite !live_statics_values.
  cbn [rs_state rs_frames state_statics].
  rewrite Hvalues, !is_some_lookup_app. intros H.
  apply orb_true_iff in H as [-> | ->]; rewrite ?orb_true_r; reflexivity.
Qed.

(* A step of a state that has the invariant and the first two parts of
   the test keeps the others, and the values of an exit may be
   evaluated. *)
Theorem region_step_calls (rs : region_state) :
  symbols_fresh rs = true ->
  cells_live rs = true ->
  locations_all_bound rs = true ->
  marks_in_variables rs = true ->
  frames_ordered rs = true ->
  calls_held rs (state_vars (rs_state rs)) (state_conts (rs_state rs)) ->
  calls_step (region_step rs).
Proof.
  intros Hfresh Hcells Hbound Hmarks Hord Hcalls.
  pose proof (proj1 (cells_live_parts rs) Hcells) as (_ & Hlisted & _).
  apply locations_all_bound_iff in Hbound.
  apply region_step_forms.
  - intros fault. exact I.
  - intros x kind d mark_kind body _.
    exact
      (begin_region_calls (count_uses rs) x kind mark_kind body Hfresh
         (fun l m Hin => proj2 (proj2 (Hlisted l m Hin))) Hcalls).
  - intros x kind p ended_by region body He.
    apply
      (end_region_calls (count_uses rs) x kind ended_by region body Hfresh
         Hord Hcalls).
    pose proof (symbols_fresh_expr rs Hfresh) as Hb. rewrite He in Hb.
    simpl in Hb. apply andb_true_iff in Hb as [Hb _].
    apply andb_true_iff in Hb as [Hb _]. exact Hb.
  - intros x kind d body y region body' _ _ _ Hcell.
    exact (local_allocation_calls (count_uses rs) x y kind d region body'
             Hfresh Hcell Hcalls).
  - intros x kind d body y region body' _ _ _ Hcell.
    exact (local_cell_allocation_calls (count_uses rs) x y d region body'
             Hbound Hcell Hcalls).
  - intros x kind d body He Husable.
    destruct
      (let_usable_no_mark (count_uses rs) x kind d body Hfresh Hmarks He
         Husable)
      as [Hsimples Hlet].
    exact
      (plain_step_calls (count_uses rs) Hfresh Hmarks Hcalls Hsimples Hlet).
  - intros bindings body He _.
    exact (static_step_calls (count_uses rs) bindings body He Hcalls).
  - intros ac He Hmark Hfault.
    apply (region_apply_cont_calls (count_uses rs) ac Hfresh Hmarks Hcalls He).
    exact (apply_cont_no_handler_usable rs ac He Hfault).
  - intros Hnot_let _ Husable.
    apply (plain_step_calls (count_uses rs) Hfresh Hmarks Hcalls Husable).
    intros x kind d body v hp' rd' He.
    contradiction (Hnot_let x kind d body He).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* With no frame, no value holds a mark. *)
Lemma value_no_mark_no_frame (v : value) : value_no_mark [] v = true.
Proof.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    simpl; try reflexivity;
    try (apply forallb_forall; intros [vslot w] Hin;
         exact (proj1 (Forall_forall _ slots) IH (vslot, w) Hin));
    try (apply forallb_forall; exact (proj1 (Forall_forall _ _) IH)).
  rewrite IHcallee. apply forallb_forall.
  exact (proj1 (Forall_forall _ _) IHargs).
Qed.

Theorem calls_kept_initial : calls_kept_initial_statement.
Proof.
  intros e statics base. apply calls_kept_parts.
  split; [ | split; [ reflexivity | repeat split ] ].
  apply marks_in_variables_parts.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ | split; [ reflexivity | intros l c v [] ] ].
  apply forallb_forall. intros [s allocated] _.
  rewrite allocated_no_mark_parts. apply andb_true_iff. split.
  - apply forallb_forall. intros w _. apply value_no_mark_no_frame.
  - apply forallb_forall. intros s' _. reflexivity.
Qed.

Theorem region_step_keeps_calls : region_step_keeps_calls_statement.
Proof.
  intros rs rs' Hinv Hcalls Hstep.
  apply region_invariant_parts in Hinv as (Hfresh & _ & Hcells & Hbound).
  apply calls_kept_parts in Hcalls as (Hmarks & Hord & Hheld).
  pose proof (proj1 (cells_live_parts rs) Hcells) as (_ & _ & _ & Hbelow & _).
  pose proof (region_step_marks rs Hfresh Hbelow Hmarks) as HA.
  pose proof (region_step_calls rs Hfresh Hcells Hbound Hmarks Hord Hheld)
    as HB.
  rewrite Hstep in HA, HB. apply calls_kept_parts.
  exact
    (conj HA (conj (region_step_keeps_frames_ordered rs rs' Hstep Hord) HB)).
Qed.

Theorem region_run_keeps_calls : region_run_keeps_calls_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros rs rs' Hinv Hcalls;
    simpl.
  - intros [= <-]. exact Hcalls.
  - destruct (region_step rs) as [rs1 | k vs rs1 | fault] eqn:Hstep;
      try discriminate.
    apply IH;
      [ exact (region_step_keeps_invariant rs rs1 Hinv Hstep)
      | exact (region_step_keeps_calls rs rs1 Hinv Hcalls Hstep) ].
Qed.

Theorem held_stays_usable : held_stays_usable_statement.
Proof.
  intros e statics base fuel rs Hfresh Hrun.
  pose proof
    (region_run_keeps_calls fuel _ rs
       (region_invariant_initial e statics base Hfresh)
       (calls_kept_initial e statics base) Hrun)
    as H.
  apply calls_kept_parts in H as (_ & _ & H & _). exact H.
Qed.

(* The values that a run exits with, of a state that has the invariant
   and passes the test. *)
Lemma region_run_exit_usable (fuel : nat) :
  ∀ (rs rs' : region_state) (k : continuation) (vs : list value),
    region_invariant rs = true ->
    calls_kept rs = true ->
    region_run fuel rs = Region_exit k vs rs' ->
    forallb (value_usable rs') vs = true.
Proof.
  induction fuel as [ | fuel IH]; intros rs rs' k vs Hinv Hcalls; simpl;
    [ discriminate | ].
  destruct (region_step rs) as [rs1 | k1 vs1 rs1 | fault] eqn:Hstep;
    [ | | discriminate ].
  - apply IH;
      [ exact (region_step_keeps_invariant rs rs1 Hinv Hstep)
      | exact (region_step_keeps_calls rs rs1 Hinv Hcalls Hstep) ].
  - intros [= <- <- <-].
    apply region_invariant_parts in Hinv as (Hfresh & _ & Hcells & Hbound).
    apply calls_kept_parts in Hcalls as (Hmarks & Hord & Hheld).
    pose proof (region_step_calls rs Hfresh Hcells Hbound Hmarks Hord Hheld)
      as HB.
    rewrite Hstep in HB. exact HB.
Qed.

Theorem exit_values_usable : exit_values_usable_statement.
Proof.
  intros e statics base fuel k vs rs Hfresh.
  exact
    (region_run_exit_usable fuel _ rs k vs
       (region_invariant_initial e statics base Hfresh)
       (calls_kept_initial e statics base)).
Qed.

Lemma erased_if_usable (rs : region_state) (vs vs' : list value) :
  forallb (value_usable rs) vs = true ->
  Forall2 (value_erased_if_usable rs) vs vs' ->
  Forall2 (value_erased rs) vs vs'.
Proof.
  intros Hu H. induction H as [ | v v' vs vs' [_ Hv] _ IH]; [ constructor | ].
  simpl in Hu. apply andb_true_iff in Hu as [Hu Hus].
  constructor; [ exact (Hv (or_introl Hu)) | exact (IH Hus) ].
Qed.

Theorem erasure_exits_erased : erasure_exits_erased_statement.
Proof.
  intros e statics base fuel k vs rs Hfresh Herasable Hrun.
  destruct (erasure_exits e statics base fuel k vs rs Hfresh Herasable Hrun)
    as (vs' & hp & Hbehaves & Hrelated).
  exists vs', hp. split; [ exact Hbehaves | ].
  exact
    (erased_if_usable rs vs vs'
       (exit_values_usable e statics base fuel k vs rs Hfresh Hrun) Hrelated).
Qed.

Theorem erasure_adds_no_exit_erased : erasure_adds_no_exit_erased_statement.
Proof.
  intros e statics base k vs' hp Hfresh Herasable Hbehaves.
  destruct
    (erasure_adds_no_exit e statics base k vs' hp Hfresh Herasable Hbehaves)
    as (fuel & [Hfault | (vs & rs & Hrun & Hrelated)]);
    exists fuel; [ left; exact Hfault | right ].
  exists vs, rs. split; [ exact Hrun | ].
  exact
    (erased_if_usable rs vs vs'
       (exit_values_usable e statics base fuel k vs rs Hfresh Hrun) Hrelated).
Qed.
