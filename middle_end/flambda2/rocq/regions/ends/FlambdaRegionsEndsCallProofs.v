(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Lemmas for the proofs of the statements of FlambdaRegionsEndsSpec.v:
   a continuation is applied, and code is called. What applying a
   continuation comes to is said by [lands]: nothing, if it is
   undefined; if it is an exit through the return continuation of what
   is run, that no plain frame was drawn since what is run was
   entered; if it is a state, that the state has the property. The
   three relations of the continuations are taken apart together
   ([landings]): a return enters a handler of the caller, returns from
   the caller in its turn, or calls what is left of an
   over-application; a raise enters an exception handler of a caller
   or leaves by the exception continuation.

   The machine of regions applies a handler that is given an open
   region by a rule of its own (FlambdaRegionsSpec.region_apply_cont),
   whose test of kinds is not that of
   FlambdaSemanticsSpec.apply_cont_value. So the lemmas are of
   [apply_unkinded], which is [apply_cont_value] without its test: a
   function of these proofs, not a rule of the machine. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaRegionsSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCallsProofs FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs.
From OxCamlFlambda2 Require Import FlambdaRegionsEndsSpec
  FlambdaRegionsEndsBasicsProofs FlambdaRegionsEndsStableProofs
  FlambdaRegionsEndsHandlersProofs FlambdaRegionsEndsJumpProofs
  FlambdaRegionsEndsLetProofs.
Import ListNotations.

(* ======================================================================== *)
(* Applying a continuation, without the test of kinds                       *)
(* ======================================================================== *)

Fixpoint apply_unkinded (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) {struct c}
    : step_result :=
  match c with
  | Outer k => Exit k vs hp rd
  | Closure (Handler params metadata handler) def_vars def_conts recursive =>
    match bind_params params vs def_vars with
    | Some handler_vars =>
      Step
        (State handler handler_vars
           (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd)
    | None => Undefined
    end
  | Extra_args c extra => apply_unkinded c (vs ++ extra) statics hp rd
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    match vs with
    | [v] =>
      run_call (fun partial => apply_unkinded return_to [partial] statics hp rd)
        (shape_of_call statics v Indirect remaining remaining_arity
           return_arity)
        return_arity return_to exn_to statics hp rd
    | _ => Undefined
    end
  end.

(* Applying a continuation is undefined, or is that. *)
Lemma apply_cont_value_unkinded (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    apply_cont_value c vs statics hp rd = Undefined
    ∨ apply_cont_value c vs statics hp rd = apply_unkinded c vs statics hp rd.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to IHret exn_to IHexn];
    intros vs statics hp rd; cbn [apply_cont_value apply_unkinded].
  - destruct (kinds_eqb _ _); [ right; reflexivity | left; reflexivity ].
  - right. reflexivity.
  - exact (IH _ _ _ _).
  - destruct vs as [ | v [ | w vs]]; try (left; reflexivity).
    destruct
      (shape_of_call statics v Indirect remaining remaining_arity return_arity)
      as [[closure c vs' args_arity
          | closure c vs' remaining' args_arity remaining_arity' | partial] | ];
      cbn [run_call].
    + right. reflexivity.
    + right. reflexivity.
    + exact (IHret _ _ _ _).
    + left. reflexivity.
Qed.

Lemma apply_unkinded_extra (c : cont_value) (extra vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  apply_unkinded (with_extra_args c extra) vs statics hp rd
  = apply_unkinded c (vs ++ extra) statics hp rd.
Proof.
  destruct extra as [ | v extra]; cbn [with_extra_args apply_unkinded];
    [ rewrite app_nil_r | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* A call                                                                   *)
(* ======================================================================== *)

(* A call of code is undefined or enters the body of the code, with
   the two continuations that it is given. *)
Lemma call_code_cases (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Undefined
  ∨ ∃ callee_vars : env,
      call_code closure c vs args_arity return_arity return_to exn_to statics
        hp rd
      = Step
          (State (code_body c) callee_vars
             [(code_return_continuation c, return_to);
              (code_exn_continuation c, exn_to)]
             statics hp rd).
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ left; reflexivity | ].
  destruct (negb _); [ left; reflexivity | ].
  destruct (bind_params _ _ _) as [callee_vars | ];
    [ right; exists callee_vars; reflexivity | left; reflexivity ].
Qed.

(* The code that a call calls is bound in the static environment. *)
Definition shape_code_bound (statics : static_env) (shape : option call_shape)
    : Prop :=
  match shape with
  | Some (Full_call _ c _ _) | Some (Over_call _ c _ _ _ _) =>
    ∃ id : code_id, lookup id (static_code statics) = Some c
  | Some (Partial_call _) | None => True
  end.

Lemma shape_of_call_bound (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  shape_code_bound statics
    (shape_of_call statics callee kind vs args_arity return_arity).
Proof.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  assert (Hclosure :
            ∃ (closure : value) (kind' : call_kind) (vs' : list value)
              (args_arity' : arity),
              shape_of_closure_call statics closure kind' vs' args_arity'
                return_arity
              = Some shape).
  { destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
      as [H | (closure & applied & applied_arity & _ & _ & _ & H)].
    - exists callee, kind, vs, args_arity. exact H.
    - exists closure, Indirect, (applied ++ vs), (applied_arity ++ args_arity).
      exact H. }
  destruct Hclosure as (closure & kind' & vs' & args_arity' & Hclosure).
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hclosure)
    as (called & slots & c & vs'' & args_arity'' & _ & Hlookup & _ & _
        & Hcode).
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcode)
    as [[_ ->] | [(_ & _ & _ & ->) | (_ & _ & _ & _ & ->)]];
    cbn [shape_code_bound]; try exact I; exists called; exact Hlookup.
Qed.

(* What the check says of code. *)
Lemma ends_code_parts (cert : certificate) (opts : ends_options) (c : code) :
  ends_code cert opts c = true ->
  Pos.eqb (code_return_continuation c) (code_exn_continuation c) = false
  ∧ ends_expr cert opts
      (ends_code_state (code_return_continuation c) (code_exn_continuation c))
      (code_body c)
    = true.
Proof.
  destruct c as [params my_closure ret_c exn_c metadata body]. intros H.
  change
    (negb (Pos.eqb ret_c exn_c)
     && ends_expr cert opts (ends_code_state ret_c exn_c) body
     = true)
    in H.
  apply andb_true_iff in H. destruct H as [Hdistinct Hbody].
  split; [ apply negb_true_iff; exact Hdistinct | exact Hbody ].
Qed.

Lemma bind_params_lengths (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    bind_params params vs vars = Some vars' -> length vs = length params.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars vars';
    cbn [bind_params length]; try discriminate; [ reflexivity | ].
  intros H. rewrite (IH vs _ vars' H). reflexivity.
Qed.

Lemma eval_simples_lengths (vars : env) (simples : list simple) :
  ∀ vs : list value,
    eval_simples vars simples = Some vs -> length vs = length simples.
Proof.
  induction simples as [ | s simples IH]; intros vs; cbn.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s); [ | discriminate ].
    destruct (eval_simples vars simples) as [vs' | ]; [ | discriminate ].
    cbn. intros [= <-]. cbn. rewrite (IH vs' eq_refl). reflexivity.
Qed.

Lemma ends_region_args_length (open : list variable) (params : list variable) :
  ∀ (kinds : list flambda_kind) (l : list simple)
    (given : list (variable * variable)),
    ends_region_args open params kinds (Some l) = Some given ->
    length l = length params.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] l given H;
    cbn in H; try discriminate.
  - destruct l; [ reflexivity | discriminate ].
  - destruct l as [ | a l]; cbn in H; [ discriminate | ].
    destruct (ends_region_args open params kinds (Some l)) as [rest | ]
      eqn:Hrest; cbn in H; [ | discriminate ].
    cbn. rewrite (IH kinds l rest Hrest). reflexivity.
Qed.

Section Landing.

Variable cert : certificate.
Variable opts : ends_options.
Variable ret : continuation.
Variable bottom : symbol.

(* The state that steps. *)
Variable rs : region_state.

Hypothesis Hordered : frames_ordered rs = true.
Hypothesis Hcode : code_ends cert opts rs.

(* ======================================================================== *)
(* What applying a continuation comes to                                    *)
(* ======================================================================== *)

Definition lands (result : step_result) : Prop :=
  match result with
  | Step s' =>
    ∀ rs' : region_state,
      state_expr (rs_state rs') = state_expr s' ->
      state_vars (rs_state rs') = state_vars s' ->
      state_conts (rs_state rs') = state_conts s' ->
      static_code (state_statics (rs_state rs'))
      = static_code (state_statics (rs_state rs)) ->
      rs_frames rs' = rs_frames rs ->
      rs_next rs' = rs_next rs ->
      state_ends cert opts ret bottom rs'
  | Exit k _ _ _ =>
    k = ret ->
    ∀ m : mark, In m (plain_marks (rs_frames rs)) -> Pos.ltb m bottom = true
  | Undefined => True
  end.

(* A caller whose open regions account for every plain frame drawn
   since its code was entered: one that is running, or one that the
   code it called has returned to. *)
Definition caller_rests (a : ends_caller) : Prop :=
  Pos.leb bottom (ec_floor a) = true
  ∧ Pos.leb (ec_floor a) (rs_next rs) = true
  ∧ ends_placed (ec_vars a) (rs_frames rs) (ec_state a) (ec_marks a)
      (ec_floor a)
  ∧ frames_tight (rs_frames rs) (ec_floor a)
      (stale_stop (es_stale (ec_state a)) (ec_floor a) (ec_marks a))
      (ec_marks a).

(* It is the caller of code that is entered now. *)
Lemma rests_called_from (a : ends_caller) (rs' : region_state)
    (u : ends_user) :
  caller_rests a ->
  rs_frames rs' = rs_frames rs ->
  eu_floor u = rs_next rs ->
  ends_called_from rs' u a.
Proof.
  intros (Hbottom & Hnext & Hplaced & Htight) Hframes Hfloor.
  unfold ends_called_from. rewrite Hframes, Hfloor.
  pose proof Hplaced as (_ & Hmp & _).
  split; [ exact Hplaced | ]. split.
  - apply forallb_forall. intros m Hm. apply Pos.ltb_lt.
    exact (placed_below_next rs (ec_marks a) m Hordered Hmp Hm).
  - split; [ exact Hnext | ].
    apply (frames_tight_range _ _ _ _ _ _ Htight); [ apply Pos.le_refl | ].
    intros m Hm.
    destruct (stale_stop (es_stale (ec_state a)) (ec_floor a) (ec_marks a))
      as [stop | ]; [ | reflexivity ].
    cbn [before] in *. apply Pos.ltb_lt in Hm. apply Pos.ltb_lt. lia.
Qed.

(* The state in which the body of code is entered has the property. *)
Lemma call_entered (a : ends_caller) (c : code)
    (return_to exn_to : cont_value) (rs' : region_state) :
  caller_rests a ->
  ends_code cert opts c = true ->
  ret_ends cert opts ret bottom rs a return_to ->
  raise_ends cert opts ret bottom rs a exn_to ->
  state_expr (rs_state rs') = code_body c ->
  state_conts (rs_state rs')
  = [(code_return_continuation c, return_to);
     (code_exn_continuation c, exn_to)] ->
  static_code (state_statics (rs_state rs'))
  = static_code (state_statics (rs_state rs)) ->
  rs_frames rs' = rs_frames rs ->
  rs_next rs' = rs_next rs ->
  state_ends cert opts ret bottom rs'.
Proof.
  intros Hrests Hc Hret Hraise Hexpr' Hconts' Hstatics' Hframes' Hnext'.
  pose proof Hrests as (Hbottom & Hnext & Hplaced & Htight).
  destruct (ends_code_parts cert opts c Hc) as [Hdistinct Hbody].
  set (ret_c := code_return_continuation c) in *.
  set (exn_c := code_exn_continuation c) in *.
  assert (Hcalled :
            ends_called_from rs'
              (Ends_user (state_vars (rs_state rs')) [] [] (rs_next rs)) a).
  { exact
      (rests_called_from a rs'
         (Ends_user (state_vars (rs_state rs')) [] [] (rs_next rs)) Hrests
         Hframes' eq_refl). }
  assert (Hkept :
            ∀ floor : symbol,
              frames_before floor (rs_frames rs')
              = frames_before floor (rs_frames rs)).
  { intros floor. rewrite Hframes'. reflexivity. }
  exists (ends_code_state ret_c exn_c).
  split; [ unfold frames_ordered; rewrite Hnext', Hframes'; exact Hordered | ].
  split; [ exact (code_ends_same cert opts rs rs' Hstatics' Hcode) | ].
  exists [], (rs_next rs).
  split; [ rewrite Hexpr'; exact Hbody | ].
  split.
  { apply Pos.leb_le. apply Pos.leb_le in Hbottom. apply Pos.leb_le in Hnext.
    lia. }
  split; [ rewrite Hnext'; apply Pos.leb_refl | ].
  split.
  - split.
    + unfold ends_placed. cbn [ends_code_state es_open es_tries es_regions].
      split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ]. intros r m [].
    + cbn [ends_code_state es_stale stale_stop]. intros m Hplain Hfloor _.
      exfalso. rewrite Hframes' in Hplain. apply plain_marks_in in Hplain.
      destruct Hplain as (g & Hg & _ & <-).
      pose proof
        (frames_ordered_below_marks (rs_frames rs) (rs_next rs) g Hordered Hg).
      apply Pos.leb_le in Hfloor. lia.
  - cbn [ends_code_state es_conts es_regions es_tries]. rewrite Hconts'.
    split.
    + intros k ec Hlookup. unfold resolve. cbn [lookup] in *.
      destruct (Pos.eqb k ret_c) eqn:Hk.
      * injection Hlookup as <-.
        apply (Ends_ok_return cert opts ret bottom rs' _ a return_to Hbottom
                 Hcalled).
        exact
          (kept_ret_ends cert opts ret bottom rs rs' a return_to (Hkept _)
             Hret).
      * destruct (Pos.eqb k exn_c) eqn:Hk'; [ | discriminate ].
        injection Hlookup as <-.
        apply (Ends_ok_raise cert opts ret bottom rs' _ a exn_to Hbottom
                 Hcalled).
        exact
          (kept_raise_ends cert opts ret bottom rs rs' a exn_to (Hkept _)
             Hraise).
    + intros k Hlookup. cbn [lookup] in *.
      destruct (Pos.eqb k ret_c); [ discriminate | ].
      destruct (Pos.eqb k exn_c); [ discriminate | ]. reflexivity.
Qed.

(* A call that is run: undefined, the body of code entered, or a
   partial application given to what [return_partial] does with it. *)
Lemma call_lands (a : ends_caller) (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (live : static_env) (hp : heap)
    (rd : read_record) :
  caller_rests a ->
  ret_ends cert opts ret bottom rs a return_to ->
  raise_ends cert opts ret bottom rs a exn_to ->
  static_code live = static_code (state_statics (rs_state rs)) ->
  shape_code_bound live shape ->
  (∀ partial : value, lands (return_partial partial)) ->
  lands
    (run_call return_partial shape return_arity return_to exn_to live hp rd).
Proof.
  intros Hrests Hret Hraise Hlive Hbound Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    cbn [run_call shape_code_bound] in *.
  - destruct Hbound as [id Hid]. rewrite Hlive in Hid.
    destruct
      (call_code_cases closure c vs args_arity return_arity return_to exn_to
         live hp rd)
      as [-> | [callee_vars ->]]; [ exact I | ].
    cbn [lands]. intros rs' Hexpr' Hvars' Hconts' Hstatics' Hframes' Hnext'.
    cbn [state_expr state_vars state_conts] in *.
    exact
      (call_entered a c return_to exn_to rs' Hrests (Hcode id c Hid) Hret
         Hraise Hexpr' Hconts' Hstatics' Hframes' Hnext').
  - destruct Hbound as [id Hid]. rewrite Hlive in Hid.
    destruct
      (call_code_cases closure c vs args_arity (code_result_arity c)
         (Over_application remaining remaining_arity return_arity return_to
            exn_to)
         exn_to live hp rd)
      as [-> | [callee_vars ->]]; [ exact I | ].
    cbn [lands]. intros rs' Hexpr' Hvars' Hconts' Hstatics' Hframes' Hnext'.
    cbn [state_expr state_vars state_conts] in *.
    exact
      (call_entered a c _ exn_to rs' Hrests (Hcode id c Hid)
         (Ends_ret_over cert opts ret bottom rs a remaining remaining_arity
            return_arity return_to exn_to Hret Hraise)
         Hraise Hexpr' Hconts' Hstatics' Hframes' Hnext').
  - exact (Hpartial partial).
  - exact I.
Qed.

(* ======================================================================== *)
(* A handler of a caller is entered                                         *)
(* ======================================================================== *)

Lemma closure_lands (a : ends_caller) (params : list variable)
    (kinds : list flambda_kind) (entry : handler_entry) (is_exn : bool)
    (c : cont_value) (args : option (list simple)) (vs : list value)
    (named : list variable) (ms_h : list mark) (live : static_env)
    (hp : heap) (rd : read_record) :
  Pos.leb bottom (ec_floor a) = true ->
  Pos.leb (ec_floor a) (rs_next rs) = true ->
  ends_placed (ec_vars a) (rs_frames rs) (ec_state a) (ec_marks a)
    (ec_floor a) ->
  cont_ends cert opts ret bottom rs (ends_caller_user a)
    (Ends_handler params kinds entry is_exn) c ->
  entry_at_jump (es_open (ec_state a)) params kinds entry args = Some named ->
  (length vs = length params -> args_given (ec_vars a) args vs) ->
  open_marks (ec_vars a) named = Some ms_h ->
  marks_placed (rs_frames rs) ms_h = true ->
  (∀ r : variable, In r named -> In r (es_open (ec_state a))) ->
  frames_tight (rs_frames rs) (ec_floor a)
    (stale_stop is_exn (ec_floor a) ms_h) ms_h ->
  lands (apply_unkinded c vs live hp rd).
Proof.
  intros Hbottom Hnext Hplaced Hcont Hnamed Hgiven Hmarks Hmp Hsub Htight.
  destruct
    (cont_ends_handler_inv cert opts ret bottom rs _ params kinds entry is_exn
       c Hcont)
    as (metadata & handler & def_vars & def_conts & recursive & es0 & base
        & -> & -> & -> & _).
  cbn [apply_unkinded].
  destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
    [ | exact I ].
  cbn [lands]. intros rs' Hexpr' Hvars' Hconts' Hstatics' Hframes' Hnext'.
  cbn [state_expr state_vars state_conts] in *.
  apply
    (handler_entered cert opts ret bottom rs' (ec_vars a) (ec_state a)
       (ec_marks a) (ec_floor a) params metadata handler def_vars def_conts
       recursive entry args vs handler_vars named ms_h);
    [ unfold frames_ordered; rewrite Hnext', Hframes'; exact Hordered
    | exact (code_ends_same cert opts rs rs' Hstatics' Hcode)
    | exact Hexpr' | exact Hvars' | exact Hconts' | exact Hbottom
    | rewrite Hnext'; exact Hnext
    | rewrite Hframes'; exact Hplaced
    | apply (kept_cont_ends cert opts ret bottom rs rs');
        [ rewrite Hframes'; reflexivity | exact Hcont ]
    | exact Hnamed
    | exact (Hgiven (bind_params_lengths _ _ _ _ Hbind))
    | exact Hbind
    | exact Hmarks
    | rewrite Hframes'; exact Hmp
    | exact Hsub
    | rewrite Hframes'; exact Htight ].
Qed.

(* ======================================================================== *)
(* A jump, a return and a raise                                             *)
(* ======================================================================== *)

(* What applying a continuation that is as its entry says comes to:
   said of the return continuation where no plain frame was drawn
   since the code of the user was entered, and of the exception
   continuation. A handler is entered through [closure_lands]. *)
Definition landing (u : ends_user) (ec : ends_cont) (c : cont_value) : Prop :=
  Pos.leb (eu_floor u) (rs_next rs) = true ->
  match ec with
  | Ends_return =>
    (∀ m : mark,
       In m (plain_marks (rs_frames rs)) -> Pos.ltb m (eu_floor u) = true) ->
    ∀ (vs : list value) (live : static_env) (hp : heap) (rd : read_record),
      static_code live = static_code (state_statics (rs_state rs)) ->
      lands (apply_unkinded c vs live hp rd)
  | Ends_raise =>
    ∀ (vs : list value) (live : static_env) (hp : heap) (rd : read_record),
      static_code live = static_code (state_statics (rs_state rs)) ->
      lands (apply_unkinded c vs live hp rd)
  | Ends_handler _ _ _ _ | Ends_unreachable => True
  end.

(* A jump that the check passes, by a caller that rests, to a
   continuation that is as its entry says. *)
Lemma jump_lands_with (a : ends_caller) (k : continuation)
    (args : option (list simple)) (vs : list value) (c : cont_value)
    (live : static_env) (hp : heap) (rd : read_record) :
  caller_rests a ->
  ends_jump (ec_state a) k args = true ->
  (∀ ec : ends_cont,
     lookup k (es_conts (ec_state a)) = Some ec ->
     cont_ends cert opts ret bottom rs (ends_caller_user a) ec c
     ∧ landing (ends_caller_user a) ec c) ->
  args_given (ec_vars a) args vs ->
  static_code live = static_code (state_statics (rs_state rs)) ->
  lands (apply_unkinded c vs live hp rd).
Proof.
  intros Hrests Hjump Hconts Hgiven Hlive.
  pose proof Hrests as (Hbottom & Hnext & Hplaced & Htight).
  unfold ends_jump in Hjump.
  destruct (lookup k (es_conts (ec_state a)))
    as [[params kinds entry is_exn | | | ] | ] eqn:Hlookup;
    try discriminate.
  - destruct (Hconts _ eq_refl) as [Hcont _].
    destruct (entry_at_jump (es_open (ec_state a)) params kinds entry args)
      as [named | ] eqn:Hnamed; [ | discriminate ].
    assert (Hmarks :
              ∃ ms_h : list mark,
                open_marks (ec_vars a) named = Some ms_h
                ∧ marks_placed (rs_frames rs) ms_h = true
                ∧ (∀ r : variable, In r named -> In r (es_open (ec_state a)))
                ∧ frames_tight (rs_frames rs) (ec_floor a)
                    (stale_stop is_exn (ec_floor a) ms_h) ms_h).
    { apply orb_true_iff in Hjump. destruct Hjump as [Hjump | Hjump].
      - apply andb_true_iff in Hjump. destruct Hjump as [-> Hjump].
        apply (oldest_marks _ _ _ _ _ _ _ Hplaced Htight); [ | exact Hjump ].
        intros m m' Hm' Hlt. pose proof Hplaced as (_ & Hmp & Hfloor & _).
        unfold marks_placed in Hmp. apply andb_true_iff in Hmp.
        exact (stale_stop_below _ _ _ m m' (proj2 Hmp) Hfloor Hm' Hlt).
      - rewrite !andb_true_iff in Hjump.
        destruct Hjump as [[Hstale Horder] Hall].
        apply negb_true_iff in Hstale. rewrite Hstale in Htight.
        cbn [stale_stop] in Htight.
        destruct
          (selected_marks _ (rs_next rs) _ _ _ _ _ Hordered Hplaced Htight
             Horder Hall)
          as (ms_h & Hms_h & Hmp_h & Hsub & Htight_h).
        exists ms_h. split; [ exact Hms_h | ]. split; [ exact Hmp_h | ].
        split; [ exact Hsub | ].
        exact (frames_tight_stale _ _ _ _ Htight_h). }
    destruct Hmarks as (ms_h & Hms_h & Hmp_h & Hsub & Htight_h).
    exact
      (closure_lands a params kinds entry is_exn c args vs named ms_h live hp
         rd Hbottom Hnext Hplaced Hcont Hnamed (fun _ => Hgiven) Hms_h Hmp_h
         Hsub Htight_h).
  - destruct (Hconts _ eq_refl) as [_ Hland].
    apply andb_true_iff in Hjump. destruct Hjump as [Hstale Hall].
    apply negb_true_iff in Hstale. rewrite Hstale in Htight.
    cbn [stale_stop] in Htight.
    apply (Hland Hnext); [ | exact Hlive ].
    exact (return_no_plain _ (rs_next rs) _ _ _ _ Hordered Hplaced Htight Hall).
  - destruct (Hconts _ eq_refl) as [_ Hland].
    exact (Hland Hnext vs live hp rd Hlive).
Qed.

(* The three relations, together. *)
Lemma landings :
  (∀ (u : ends_user) (ec : ends_cont) (c : cont_value),
     cont_ends cert opts ret bottom rs u ec c -> landing u ec c)
  ∧ (∀ (a : ends_caller) (c : cont_value),
       ret_ends cert opts ret bottom rs a c ->
       caller_rests a ->
       ∀ (vs : list value) (live : static_env) (hp : heap) (rd : read_record),
         static_code live = static_code (state_statics (rs_state rs)) ->
         lands (apply_unkinded c vs live hp rd))
  ∧ (∀ (a : ends_caller) (c : cont_value),
       raise_ends cert opts ret bottom rs a c ->
       ∀ u : ends_user,
         Pos.leb bottom (ec_floor a) = true ->
         Pos.leb (eu_floor u) (rs_next rs) = true ->
         ends_called_from rs u a ->
         ∀ (vs : list value) (live : static_env) (hp : heap)
           (rd : read_record),
           static_code live = static_code (state_statics (rs_state rs)) ->
           lands (apply_unkinded c vs live hp rd)).
Proof.
  apply (conts_ends_mutind cert opts ret bottom rs
           (fun u ec c _ => landing u ec c)
           (fun a c _ =>
              caller_rests a ->
              ∀ (vs : list value) (live : static_env) (hp : heap)
                (rd : read_record),
                static_code live = static_code (state_statics (rs_state rs)) ->
                lands (apply_unkinded c vs live hp rd))
           (fun a c _ =>
              ∀ u : ends_user,
                Pos.leb bottom (ec_floor a) = true ->
                Pos.leb (eu_floor u) (rs_next rs) = true ->
                ends_called_from rs u a ->
                ∀ (vs : list value) (live : static_env) (hp : heap)
                  (rd : read_record),
                  static_code live
                  = static_code (state_statics (rs_state rs)) ->
                  lands (apply_unkinded c vs live hp rd))).
  - intros u params metadata handler entry vars conts recursive es base
      Hmade Hexpr Hdistinct Hnew Hopen Hin Hconts IH Hnone Hnext.
    exact I.
  - intros u c Hnext. exact I.
  - intros u Hfloor Hnext Hnone vs live hp rd Hlive.
    cbn [apply_unkinded lands]. intros _ m Hm. rewrite <- Hfloor.
    exact (Hnone m Hm).
  - intros u a c Hbottom Hcalled Hret IH Hnext Hnone vs live hp rd Hlive.
    apply IH; [ | exact Hlive ].
    destruct Hcalled as (Hplaced & Hlt & Hfloor & Htight).
    split; [ exact Hbottom | ]. split.
    { apply Pos.leb_le. apply Pos.leb_le in Hfloor. apply Pos.leb_le in Hnext.
      lia. }
    split; [ exact Hplaced | ].
    intros m Hplain Hfl Hbefore. apply (Htight m Hplain Hfl).
    pose proof (Hnone m Hplain) as Hm. apply Pos.ltb_lt in Hm.
    destruct (stale_stop (es_stale (ec_state a)) (ec_floor a) (ec_marks a))
      as [stop | ]; cbn [before] in *; apply Pos.ltb_lt;
      [ apply Pos.ltb_lt in Hbefore; lia | exact Hm ].
  - intros u k extra Hk Hnext vs live hp rd Hlive.
    rewrite apply_unkinded_extra. cbn [apply_unkinded lands]. intros Heq.
    rewrite Heq, Pos.eqb_refl in Hk. discriminate.
  - intros u a c Hbottom Hcalled Hraise IH Hnext vs live hp rd Hlive.
    exact (IH u Hbottom Hnext Hcalled vs live hp rd Hlive).
  - intros a k c Hjump Hconts IH Hrests vs live hp rd Hlive.
    apply (jump_lands_with a k None vs c live hp rd Hrests Hjump);
      [ | apply args_given_none | exact Hlive ].
    intros ec Hec. split; [ exact (Hconts ec Hec) | exact (IH ec Hec) ].
  - intros a remaining remaining_arity return_arity return_to exn_to
      Hret IHret Hraise IHraise Hrests vs live hp rd Hlive.
    cbn [apply_unkinded]. destruct vs as [ | v [ | w vs]]; try exact I.
    apply
      (call_lands a _ _ return_arity return_to exn_to live hp rd Hrests Hret
         Hraise Hlive);
      [ apply shape_of_call_bound | ].
    intros partial. exact (IHret Hrests [partial] live hp rd Hlive).
  - intros a k simples extra c Heval Hraise_to Hconts IH u Hbottom Hnext
      Hcalled vs live hp rd Hlive.
    rewrite apply_unkinded_extra.
    destruct Hcalled as (Hplaced & Hlt & Hfloor & Htight).
    assert (Hnext_a : Pos.leb (ec_floor a) (rs_next rs) = true).
    { apply Pos.leb_le. apply Pos.leb_le in Hfloor. apply Pos.leb_le in Hnext.
      lia. }
    unfold ends_raise_to in Hraise_to.
    destruct (lookup k (es_conts (ec_state a)))
      as [[params kinds entry is_exn | | | ] | ] eqn:Hlookup;
      try discriminate.
    + apply andb_true_iff in Hraise_to. destruct Hraise_to as [-> Hnamed_ok].
      destruct
        (entry_at_jump (es_open (ec_state a)) params kinds entry
           (Some (unit_simple :: simples)))
        as [named | ] eqn:Hnamed; [ | discriminate ].
      pose proof Hplaced as (_ & Hmp & Hfloor_ms & _).
      unfold marks_placed in Hmp. apply andb_true_iff in Hmp.
      destruct Hmp as [_ Hdecrease].
      destruct
        (oldest_marks (rs_frames rs) (ec_vars a) (ec_state a) (ec_marks a)
           (ec_floor a) _ named Hplaced Htight)
        as (ms_h & Hms_h & Hmp_h & Hsub & Htight_h);
        [ | exact Hnamed_ok | ].
      * intros m m' Hm' Hlt'.
        pose proof
          (stale_stop_below (es_stale (ec_state a)) (ec_floor a) (ec_marks a)
             m m' Hdecrease Hfloor_ms Hm' Hlt')
          as Hbelow.
        assert (Hm_u : (m < eu_floor u)%positive).
        { destruct Hm' as [Hin | ->].
          - pose proof (proj1 (forallb_forall _ _) Hlt m' Hin) as H.
            cbn beta in H. apply Pos.ltb_lt in H. lia.
          - apply Pos.leb_le in Hfloor. lia. }
        destruct (stale_stop (es_stale (ec_state a)) (ec_floor a) (ec_marks a))
          as [stop | ]; cbn [before] in *; apply Pos.ltb_lt;
          [ apply Pos.ltb_lt in Hbelow; lia | exact Hm_u ].
      * apply
          (closure_lands a params kinds entry true c
             (Some (unit_simple :: simples)) (vs ++ extra) named ms_h live hp
             rd Hbottom Hnext_a Hplaced (proj1 (conj (Hconts _ eq_refl) I))
             Hnamed);
          [ | exact Hms_h | exact Hmp_h | exact Hsub | exact Htight_h ].
        intros Hlength. unfold entry_at_jump in Hnamed.
        destruct
          (ends_region_args (es_open (ec_state a)) params kinds
             (Some (unit_simple :: simples)))
          as [given | ] eqn:Hgiven; [ | discriminate ].
        pose proof (ends_region_args_length _ _ _ _ _ Hgiven) as Hl.
        pose proof (eval_simples_lengths _ _ _ Heval) as He.
        rewrite length_app in Hlength. cbn [length] in Hl.
        destruct vs as [ | v [ | w vs]]; cbn [length] in Hlength; try lia.
        exact (args_given_raise (ec_vars a) simples extra v Heval).
    + exact (IH _ eq_refl Hnext_a (vs ++ extra) live hp rd Hlive).
Qed.

(* A jump that the check passes, by a caller that rests whose
   continuations are as their entries say. *)
Lemma jump_lands (a : ends_caller) (k : continuation)
    (args : option (list simple)) (vs : list value) (conts : cont_env)
    (live : static_env) (hp : heap) (rd : read_record) :
  caller_rests a ->
  conts_ends cert opts ret bottom rs (ends_caller_user a)
    (es_conts (ec_state a)) conts ->
  ends_jump (ec_state a) k args = true ->
  args_given (ec_vars a) args vs ->
  static_code live = static_code (state_statics (rs_state rs)) ->
  lands (apply_unkinded (resolve k conts) vs live hp rd).
Proof.
  intros Hrests [Hbound _] Hjump Hgiven Hlive.
  apply (jump_lands_with a k args vs _ live hp rd Hrests Hjump);
    [ | exact Hgiven | exact Hlive ].
  intros ec Hec.
  split;
    [ exact (Hbound k ec Hec)
    | exact (proj1 landings _ _ _ (Hbound k ec Hec)) ].
Qed.

(* ======================================================================== *)
(* The state that steps: a jump and a call                                  *)
(* ======================================================================== *)

Lemma running_rests (es : ends_state) (ms : list mark) (floor : symbol) :
  ends_at cert opts ret bottom rs es ms floor ->
  caller_rests (Ends_caller es (state_vars (rs_state rs)) ms floor).
Proof.
  intros (_ & Hbottom & Hnext & [Hplaced Htight] & _).
  split; [ exact Hbottom | ]. split; [ exact Hnext | ].
  split; [ exact Hplaced | exact Htight ].
Qed.

Lemma apply_cont_lands (es : ends_state) (ms : list mark) (floor : symbol)
    (ac : apply_cont) (vs : list value) (live : static_env) (hp : heap)
    (rd : read_record) :
  ends_at cert opts ret bottom rs es ms floor ->
  state_expr (rs_state rs) = Apply_cont ac ->
  eval_simples (state_vars (rs_state rs)) (ac_args ac) = Some vs ->
  static_code live = static_code (state_statics (rs_state rs)) ->
  lands
    (apply_unkinded (resolve (ac_continuation ac) (state_conts (rs_state rs)))
       vs live hp rd).
Proof.
  intros Hat He Heval Hlive.
  pose proof (running_rests es ms floor Hat) as Hrests.
  destruct Hat as (Hexpr & _ & _ & _ & Hconts).
  rewrite He in Hexpr. cbn [ends_expr] in Hexpr.
  apply
    (jump_lands (Ends_caller es (state_vars (rs_state rs)) ms floor)
       (ac_continuation ac) (Some (ac_args ac)) vs _ live hp rd Hrests Hconts
       Hexpr);
    [ | exact Hlive ].
  apply args_given_eval. intros l [= <-]. exact Heval.
Qed.

Lemma apply_lands (es : ends_state) (ms : list mark) (floor : symbol)
    (ap : apply_expr) (callee : value) (vs extra : list value)
    (live : static_env) (hp : heap) (rd : read_record) :
  ends_at cert opts ret bottom rs es ms floor ->
  state_expr (rs_state rs) = Apply ap ->
  eval_simples (state_vars (rs_state rs)) (exn_extra_simples ap) = Some extra ->
  static_code live = static_code (state_statics (rs_state rs)) ->
  lands
    (call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
       (ap_return_arity ap)
       (resolve (ap_continuation ap) (state_conts (rs_state rs)))
       (with_extra_args
          (resolve (ap_exn_continuation ap) (state_conts (rs_state rs))) extra)
       live hp rd).
Proof.
  intros Hat He Hextra Hlive.
  pose proof (running_rests es ms floor Hat) as Hrests.
  destruct Hat as (Hexpr & _ & _ & _ & [Hbound _]).
  rewrite He in Hexpr. cbn [ends_expr] in Hexpr.
  apply andb_true_iff in Hexpr. destruct Hexpr as [Hjump Hraise].
  set (a := Ends_caller es (state_vars (rs_state rs)) ms floor) in *.
  assert (Hret :
            ret_ends cert opts ret bottom rs a
              (resolve (ap_continuation ap) (state_conts (rs_state rs)))).
  { apply (Ends_ret_to cert opts ret bottom rs a (ap_continuation ap) _ Hjump).
    intros ec Hec. exact (Hbound _ ec Hec). }
  assert (Hraise' :
            raise_ends cert opts ret bottom rs a
              (with_extra_args
                 (resolve (ap_exn_continuation ap) (state_conts (rs_state rs)))
                 extra)).
  { apply
      (Ends_raise_to cert opts ret bottom rs a (ap_exn_continuation ap)
         (exn_extra_simples ap) extra _ Hextra Hraise).
    intros ec Hec. exact (Hbound _ ec Hec). }
  unfold call_function.
  apply
    (call_lands a _ _ (ap_return_arity ap) _ _ live hp rd Hrests Hret Hraise'
       Hlive);
    [ apply shape_of_call_bound | ].
  intros partial.
  destruct
    (apply_cont_value_unkinded
       (resolve (ap_continuation ap) (state_conts (rs_state rs))) [partial]
       live hp rd)
    as [-> | ->]; [ exact I | ].
  exact (proj1 (proj2 landings) a _ Hret Hrests [partial] live hp rd Hlive).
Qed.

(* The step of FlambdaSemanticsSpec.step of a call or of a jump. *)
Lemma step_lands (es : ends_state) (ms : list mark) (floor : symbol)
    (live : static_env) :
  ends_at cert opts ret bottom rs es ms floor ->
  static_code live = static_code (state_statics (rs_state rs)) ->
  (∃ ap : apply_expr, state_expr (rs_state rs) = Apply ap)
  ∨ (∃ ac : apply_cont, state_expr (rs_state rs) = Apply_cont ac) ->
  lands (step (with_statics (rs_state rs) live)).
Proof.
  intros Hat Hlive Hform. unfold with_statics.
  destruct Hform as [[ap He] | [ac He]]; rewrite He;
    autorewrite with step.
  - destruct (eval_simple (state_vars (rs_state rs)) (ap_callee ap))
      as [callee | ]; [ | exact I ].
    destruct (eval_simples (state_vars (rs_state rs)) (ap_args ap))
      as [vs | ]; [ | exact I ].
    destruct (eval_simples (state_vars (rs_state rs)) (exn_extra_simples ap))
      as [extra | ] eqn:Hextra; [ | exact I ].
    exact (apply_lands es ms floor ap callee vs extra live _ _ Hat He Hextra
             Hlive).
  - destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac))
      as [vs | ] eqn:Heval; [ | exact I ].
    unfold apply_continuation.
    destruct
      (apply_cont_value_unkinded
         (resolve (ac_continuation ac) (state_conts (rs_state rs))) vs live
         (state_heap (rs_state rs)) (state_reads (rs_state rs)))
      as [-> | ->]; [ exact I | ].
    exact (apply_cont_lands es ms floor ac vs live _ _ Hat He Heval Hlive).
Qed.

End Landing.

(* An exit is of a call or of a jump. *)
Lemma step_exit_forms (s : state) (k : continuation) (vs : list value)
    (hp : heap) (rd : read_record) :
  step s = Exit k vs hp rd ->
  (∃ ap : apply_expr, state_expr s = Apply ap)
  ∨ (∃ ac : apply_cont, state_expr s = Apply_cont ac).
Proof.
  destruct s as [e vars conts statics hp0 rd0].
  destruct e as [x kind d body | bindings body | k0 h body | handlers body
                | ap | ac | scrutinee arms | ];
    autorewrite with step; cbn [state_expr]; intros Hstep.
  - destruct (exec_named statics hp0 rd0 vars d) as [[[v hp'] rd'] | ];
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); discriminate.
  - destruct (eval_static_bindings vars bindings statics); discriminate.
  - discriminate.
  - destruct (handlers_bind_statics handlers); discriminate.
  - left. exists ap. reflexivity.
  - right. exists ac. reflexivity.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms); discriminate.
  - discriminate.
Qed.
