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

(* Proofs of the statements of FlambdaKnownSymbolsSpec.v. Nothing here
   needs review.

   The unit and the unit folded are run side by side, in step, with the
   same values. Two things are kept along the run of the unit: its
   terms bind the known symbols as the unit says
   (FlambdaClosuresProofs.terms_inv_step), so what the run has bound of
   them is what is known ([agrees]). A defining expression that has a
   value in such a static environment has the same value in the known
   one, by the laws of the primitives about a static environment that
   grows (FlambdaPrimitiveProofs.eval_prim_statics_extend).

   The two runs may start from a continuation environment of what may
   follow a unit (FlambdaObserversSpec.observers). What follows is the
   same term in the two runs, and the folding is not the identity on
   it: a primitive applied to constants is folded wherever it is. So
   the second state is not the first with every term folded. It is
   given by a relation ([state_kept]): each term that the first state
   keeps, its expression and the handlers of its continuations, is
   folded in the second or, if it binds nothing static, left as it is;
   the code is folded. A step is answered for the two ways
   ([kept_step_components]). For a unit run on its own nothing is left
   as it is. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaNormaliseSpec
  FlambdaClosuresProofs FlambdaUnusedStaticsSpec FlambdaUnusedSlotsProofs
  FlambdaWellFormedSpec FlambdaImportsSpec FlambdaObserversSpec
  FlambdaUnitFromProofs FlambdaKnownAritySpec FlambdaKnownArityProofs
  FlambdaKnownSymbolsSpec FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* What has no variable in it                                               *)
(* ======================================================================== *)

Lemma eval_closed_simple (vars : env) (s : simple) :
  closed_simple s = true -> eval_simple vars s = eval_simple [] s.
Proof. destruct s; simpl; [ discriminate | reflexivity | reflexivity ]. Qed.

Lemma eval_closed_simples (vars : env) (args : list simple) :
  forallb closed_simple args = true ->
  eval_simples vars args = eval_simples [] args.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hs Hargs].
  rewrite (eval_closed_simple vars s Hs), (IH Hargs). reflexivity.
Qed.

Lemma eval_closed_value_slots (vars : env)
    (slots : list (value_slot * simple)) :
  forallb (fun slot => closed_simple (snd slot)) slots = true ->
  eval_value_slots vars slots = eval_value_slots [] slots.
Proof.
  induction slots as [ | [slot s] slots IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hs Hslots].
  rewrite (eval_closed_simple vars s Hs), (IH Hslots). reflexivity.
Qed.

Lemma eval_closed_static_const (vars : env) (const : static_const) :
  closed_static_const const = true ->
  eval_static_const vars const = eval_static_const [] const.
Proof.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl; intros H.
  - rewrite (eval_closed_value_slots vars slots H). reflexivity.
  - rewrite (eval_closed_value_slots vars slots H). reflexivity.
  - rewrite (eval_closed_simples vars fields H). reflexivity.
  - rewrite (eval_closed_simple vars contents H). reflexivity.
  - reflexivity.
  - rewrite (eval_closed_simples vars elements H). reflexivity.
Qed.

(* ======================================================================== *)
(* A defining expression is folded to its value                             *)
(* ======================================================================== *)

Section Folded.

Variables statics run_statics : static_env.

(* Each symbol that [statics] binds is bound in [run_statics] to the same
   thing or to nothing. *)
Hypothesis Hagrees :
  ∀ (s : symbol) (allocated allocated' : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    lookup s (static_values run_statics) = Some allocated' ->
    allocated' = allocated.

Lemma known_symbol_bound (s : symbol)
    (allocated : statically_allocated_value) :
  is_some (lookup s (static_values statics)) = true ->
  lookup s (static_values run_statics) = Some allocated ->
  lookup s (static_values statics) = Some allocated.
Proof.
  destruct (lookup s (static_values statics)) as [known | ] eqn:Hs;
    [ | discriminate ].
  intros _ Hrun. rewrite (Hagrees s known allocated Hs Hrun). reflexivity.
Qed.

Lemma eval_known_simple (vars : env) (s : simple) :
  simple_known statics s = true -> eval_simple vars s = eval_simple [] s.
Proof. destruct s; simpl; [ discriminate | reflexivity | reflexivity ]. Qed.

Lemma eval_known_simples (vars : env) (args : list simple) :
  forallb (simple_known statics) args = true ->
  eval_simples vars args = eval_simples [] args.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hs Hargs].
  rewrite (eval_known_simple vars s Hs), (IH Hargs). reflexivity.
Qed.

(* The symbols among the values of known arguments are bound. *)
Lemma known_simples_symbols (args : list simple) :
  forallb (simple_known statics) args = true ->
  ∀ (vs : list value) (s : symbol),
    eval_simples [] args = Some vs ->
    Value_symbol s ∈ vs ->
    is_some (lookup s (static_values statics)) = true.
Proof.
  induction args as [ | a args IH]; simpl; intros H vs s.
  - intros [= <-] [].
  - apply andb_true_iff in H as [Ha Hargs].
    destruct (eval_simple [] a) as [v | ] eqn:Hv; simpl; [ | discriminate ].
    destruct (eval_simples [] args) as [rest | ]; simpl; [ | discriminate ].
    intros [= <-] [Hhead | Hrest]; [ | exact (IH Hargs rest s eq_refl Hrest) ].
    subst v. destruct a as [x | sym | c]; simpl in Hv, Ha;
      [ discriminate | | discriminate ].
    injection Hv as <-. exact Ha.
Qed.

(* What is read of a known symbol in the run is read of it in what is
   known. *)
Lemma eval_known_named (vars : env) (d : named) (v : value) :
  named_known statics d = true ->
  eval_named run_statics vars d = Some v ->
  eval_named statics [] d = Some v.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl;
    try discriminate.
  - intros H. apply andb_true_iff in H as [_ Hargs].
    rewrite (eval_known_simples vars args Hargs).
    destruct (eval_simples [] args) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    apply eval_prim_statics_extend. intros s allocated Hin.
    apply known_symbol_bound.
    exact (known_simples_symbols args Hargs vs s Hvs Hin).
  - intros Hclosure. rewrite (eval_known_simple vars closure Hclosure).
    destruct closure as [x | s | c]; simpl in Hclosure |- *;
      [ discriminate | | discriminate ].
    destruct (lookup s (static_values run_statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    rewrite (known_symbol_bound s allocated Hclosure Hs). exact (fun H => H).
  - intros Hclosure. rewrite (eval_known_simple vars closure Hclosure).
    destruct closure as [x | s | c]; simpl in Hclosure |- *;
      [ discriminate | | discriminate ].
    unfold project_value_slot, closure_in_set.
    destruct (lookup s (static_values run_statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    rewrite (known_symbol_bound s allocated Hclosure Hs). exact (fun H => H).
  - intros Hclosure. rewrite (eval_known_simple vars closure Hclosure).
    destruct closure as [x | s | c]; simpl in Hclosure |- *;
      [ discriminate | | discriminate ].
    destruct (lookup s (static_values run_statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    rewrite (known_symbol_bound s allocated Hclosure Hs). exact (fun H => H).
Qed.

Lemma fold_known_named_sound_here (vars : env) (d : named) (v : value) :
  eval_named run_statics vars d = Some v ->
  eval_named run_statics vars (fold_known_named statics d) = Some v.
Proof.
  intros Hv. unfold fold_known_named.
  destruct (named_known statics d) eqn:Hknown; [ | exact Hv ].
  rewrite (eval_known_named vars d v Hknown Hv).
  destruct v; solve [ exact Hv | reflexivity ].
Qed.

End Folded.

Theorem fold_known_named_sound : fold_known_named_sound_statement.
Proof.
  intros statics run_statics vars d v Hagrees.
  exact (fold_known_named_sound_here statics run_statics Hagrees vars d v).
Qed.

(* ======================================================================== *)
(* The projection of a function slot is folded to its value                 *)
(* ======================================================================== *)

Section Folded_slot.

Variables sets run_statics : static_env.
Variable hp : heap.
Variable rd : read_record.

(* Each symbol that [sets] binds is bound in [run_statics] to a closure
   at the same slot of the same set, or to nothing. *)
Hypothesis Hsets :
  ∀ (s : symbol) (allocated allocated' : statically_allocated_value),
    lookup s (static_values sets) = Some allocated ->
    lookup s (static_values run_statics) = Some allocated' ->
    same_set allocated allocated'.

Lemma fold_function_slot_sound_here (vars : env) (d : named) (v : value) :
  eval_named run_statics vars d = Some v ->
  eval_named run_statics vars (fold_function_slot sets d) = Some v.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact (fun H => H).
  destruct closure as [x | s | c]; try exact (fun H => H).
  unfold fold_function_slot.
  change
    (eval_named sets [] (Project_function_slot move_from move_to (Symbol s)))
    with (project_function_slot sets move_from move_to (Value_symbol s)).
  change
    (eval_named run_statics vars
       (Project_function_slot move_from move_to (Symbol s)))
    with (project_function_slot run_statics move_from move_to (Value_symbol s))
    at 1.
  intros Hrun.
  destruct (project_function_slot sets move_from move_to (Value_symbol s))
    as [w | ] eqn:Hknown;
    [ | exact Hrun ].
  destruct w; try exact Hrun.
  change (Some (Value_symbol s0) = Some v).
  unfold project_function_slot in Hrun, Hknown.
  destruct (lookup s (static_values sets)) as [allocated | ] eqn:Hs;
    [ | discriminate ].
  destruct (lookup s (static_values run_statics)) as [allocated' | ] eqn:Hs';
    [ | discriminate ].
  pose proof (Hsets s allocated allocated' Hs Hs') as Hsame.
  destruct allocated as [code slots | fslot syms decls slots | | | | | ],
    allocated' as [code' slots' | fslot' syms' decls' slots' | | | | | ];
    try contradiction.
  - rewrite Hknown in Hrun. exact Hrun.
  - destruct Hsame as (-> & -> & ->). rewrite Hknown in Hrun. exact Hrun.
  - discriminate Hknown.
Qed.

(* A symbol that [sets] binds to a closure of code is, in the run, a
   closure of that code, or bound to nothing. *)
Lemma closure_of_sets (s : symbol) (code called : code_id)
    (slots slots' : list (value_slot * value)) :
  closure_of sets (Value_symbol s) = Some (code, slots) ->
  closure_of run_statics (Value_symbol s) = Some (called, slots') ->
  called = code.
Proof.
  unfold closure_of.
  destruct (lookup s (static_values sets)) as [allocated | ] eqn:Hs;
    [ | discriminate ].
  destruct (lookup s (static_values run_statics)) as [allocated' | ] eqn:Hs';
    [ | discriminate ].
  pose proof (Hsets s allocated allocated' Hs Hs') as Hsame.
  destruct allocated as [c sl | fslot syms decls sl | | | | | ],
    allocated' as [c' sl' | fslot' syms' decls' sl' | | | | | ];
    try contradiction.
  - simpl in Hsame. subst c'. intros [= <- _] [= <- _]. reflexivity.
  - destruct Hsame as (-> & -> & ->).
    destruct (lookup fslot decls) as [c | ]; simpl; [ | discriminate ].
    intros [= <- _] [= <- _]. reflexivity.
  - discriminate.
Qed.

(* The fields of two blocks, the second with each field of the first
   that is known: where the first has a field, the second has one, the
   same if that of the first is known. *)
Lemma known_field_nth (fields fields' : list value) (n : nat) (known : value) :
  Forall2
    (fun known actual => known = Value_const unknown_field ∨ actual = known)
    fields fields' ->
  nth_error fields n = Some known ->
  ∃ actual : value,
    nth_error fields' n = Some actual
    ∧ (known = Value_const unknown_field ∨ actual = known).
Proof.
  intros H. revert n.
  induction H as [ | k a fields fields' Hfirst _ IH]; intros [ | n]; simpl;
    try discriminate.
  - intros [= <-]. exists a. split; [ reflexivity | exact Hfirst ].
  - apply IH.
Qed.

(* The load of a field of a block that [sets] binds has, in the run,
   the value that it has under [sets], or none. *)
Lemma eval_block_load_sets (kind : block_access_kind) (field : Z)
    (s : symbol) (w v : value) :
  eval_block_load sets kind field (Value_symbol s) = Some w ->
  eval_block_load run_statics kind field (Value_symbol s) = Some v ->
  v = w.
Proof.
  destruct kind as [tag size field_kind]. unfold eval_block_load, block_of.
  destruct (lookup s (static_values sets)) as [allocated | ] eqn:Hs;
    [ | discriminate ].
  destruct (lookup s (static_values run_statics)) as [allocated' | ] eqn:Hs';
    [ | discriminate ].
  pose proof (Hsets s allocated allocated' Hs Hs') as Hsame.
  destruct allocated as [c sl | fslot syms decls sl | btag fields | | | | ];
    try discriminate.
  destruct allocated' as [c' sl' | fslot' syms' decls' sl' | btag' fields'
                         | | | | ];
    try contradiction.
  destruct Hsame as [-> Hfields]. simpl.
  rewrite <- (Forall2_length Hfields).
  destruct
    (known_is tag btag && known_is size (Z.of_nat (length fields))
     && (0 <=? field)%Z);
    [ | discriminate ].
  destruct (nth_error fields (Z.to_nat field)) as [known | ] eqn:Hknown;
    simpl; [ | discriminate ].
  destruct (known_field_nth fields fields' _ known Hfields Hknown)
    as (actual & -> & Hactual).
  simpl. intros Hw.
  destruct Hactual as [-> | ->].
  - destruct field_kind; discriminate Hw.
  - destruct
      (match field_kind with
       | Any_value => kind_eqb (kind_of_value known) Value
       | Immediate => is_tagged_immediate known
       end);
      [ | discriminate Hw ].
    injection Hw as <-. intros [= <-]. reflexivity.
Qed.

Lemma fold_block_load_sound_here (vars : env) (d : named) (v : value) :
  eval_named run_statics vars d = Some v ->
  eval_named run_statics vars (fold_block_load sets d) = Some v.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact (fun H => H).
  destruct p as [p | p | p | p | rp]; try exact (fun H => H).
  destruct p; try exact (fun H => H).
  destruct args as [ | [x | s | c] [ | arg args]]; try exact (fun H => H);
    try (destruct mut; exact (fun H => H)).
  set (load := Prim (Unary (Block_load kind mut field)) [Symbol s]).
  assert (Hload :
    eval_named run_statics vars load = Some v ->
    eval_named run_statics vars
      match eval_named sets [] load with
      | Some (Value_const c) => Simple (Const c)
      | Some (Value_symbol s') => Simple (Symbol s')
      | _ => load
      end
    = Some v).
  { intros Hrun.
    destruct (eval_named sets [] load) as [w | ] eqn:Hknown;
      [ | exact Hrun ].
    unfold load in Hknown, Hrun.
    assert (Hv : v = w).
    { cbn [eval_named eval_simples eval_simple] in Hknown, Hrun.
      simpl in Hknown, Hrun.
      rewrite eval_prim_unary in Hknown, Hrun.
      exact (eval_block_load_sets kind field s w v Hknown Hrun). }
    subst w. destruct v; solve [ exact Hrun | reflexivity ]. }
  destruct mut; [ exact (fun H => H) | exact Hload | exact Hload ].
Qed.

(* A call of unknown arity of such a symbol and the direct call of that
   code step alike: both are undefined where the symbol is bound to
   nothing. *)
Lemma fold_known_apply_unknown_arity_here (vars : env) (conts : cont_env)
    (ap : apply_expr) :
  (∀ code_ids : option (list code_id),
    ap_call_kind ap <> Indirect_known_arity code_ids) ->
  step (State (Apply (fold_known_apply sets ap)) vars conts run_statics hp rd)
  = step (State (Apply ap) vars conts run_statics hp rd).
Proof.
  intros Hnot. unfold fold_known_apply.
  destruct (ap_call_kind ap) as [named | | code_ids] eqn:Hkind;
    [ reflexivity | | destruct (Hnot code_ids eq_refl) ].
  destruct (ap_callee ap) as [x | s | c] eqn:Hcallee; try reflexivity.
  destruct (closure_of sets (Value_symbol s)) as [[code slots] | ] eqn:Hknown;
    [ | reflexivity ].
  rewrite !step_apply. unfold exn_extra_simples.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  rewrite Hkind, Hcallee. cbn [eval_simple].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (map fst (ap_exn_extra_args ap)))
    as [extra | ];
    [ | reflexivity ].
  unfold call_function.
  rewrite
    !(shape_of_call_closure run_statics (Value_symbol s) _ vs _ _ eq_refl).
  destruct (closure_of run_statics (Value_symbol s))
    as [[called slots'] | ] eqn:Hrun.
  - rewrite
      (shape_of_closure_call_kind run_statics (Value_symbol s) (Direct code)
         vs _ _ called slots' Hrun
         (or_intror
            (f_equal Direct
               (eq_sym (closure_of_sets s code called slots slots' Hknown
                          Hrun))))).
    reflexivity.
  - unfold shape_of_closure_call. rewrite Hrun. reflexivity.
Qed.

(* A call of known arity is folded as the same call of unknown arity
   is, or is left. *)
Lemma fold_known_apply_known_arity (ap : apply_expr)
    (code_ids : option (list code_id)) :
  ap_call_kind ap = Indirect_known_arity code_ids ->
  fold_known_apply sets ap
  = fold_known_apply sets (apply_with_kind Indirect_unknown_arity ap)
  ∨ fold_known_apply sets ap = ap.
Proof.
  destruct ap as [callee kind args k exn extra args_arity return_arity].
  cbn [ap_call_kind]. intros ->.
  unfold fold_known_apply, apply_with_kind.
  cbn [ap_call_kind ap_callee ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  destruct callee as [x | s | c]; [ right; reflexivity | | right; reflexivity ].
  destruct (closure_of sets (Value_symbol s)) as [[code slots] | ];
    [ left; reflexivity | right; reflexivity ].
Qed.

(* A call that has a step is folded to a call with that step. A call
   of known arity that has a step is the call of unknown arity
   (FlambdaKnownArityProofs.known_arity_step), which is folded to the
   same direct call. *)
Lemma fold_known_apply_sound_here (vars : env) (conts : cont_env)
    (ap : apply_expr) :
  step (State (Apply ap) vars conts run_statics hp rd) <> Undefined ->
  step (State (Apply (fold_known_apply sets ap)) vars conts run_statics hp rd)
  = step (State (Apply ap) vars conts run_statics hp rd).
Proof.
  intros Hdefined.
  destruct (ap_call_kind ap) as [named | | code_ids] eqn:Hkind.
  - apply fold_known_apply_unknown_arity_here.
    intros ids Hids. rewrite Hkind in Hids. discriminate Hids.
  - apply fold_known_apply_unknown_arity_here.
    intros ids Hids. rewrite Hkind in Hids. discriminate Hids.
  - destruct (fold_known_apply_known_arity ap code_ids Hkind) as [-> | ->];
      [ | reflexivity ].
    rewrite fold_known_apply_unknown_arity_here
      by (intros ids Hids; destruct ap; discriminate Hids).
    exact (known_arity_step ap code_ids vars conts run_statics hp rd Hkind
             Hdefined).
Qed.

End Folded_slot.

Theorem fold_function_slot_sound : fold_function_slot_sound_statement.
Proof.
  intros sets run_statics vars d v Hsets.
  exact (fold_function_slot_sound_here sets run_statics Hsets vars d v).
Qed.

Theorem fold_block_load_sound : fold_block_load_sound_statement.
Proof.
  intros sets run_statics vars d v Hsets.
  exact (fold_block_load_sound_here sets run_statics Hsets vars d v).
Qed.

Theorem fold_known_apply_sound : fold_known_apply_sound_statement.
Proof.
  intros sets run_statics hp rd vars conts ap Hsets.
  exact
    (fold_known_apply_sound_here sets run_statics hp rd Hsets vars conts ap).
Qed.

Theorem fold_known_apply_unknown_arity :
  fold_known_apply_unknown_arity_statement.
Proof.
  intros sets run_statics hp rd vars conts ap Hsets.
  exact
    (fold_known_apply_unknown_arity_here sets run_statics hp rd Hsets vars
       conts ap).
Qed.

(* ======================================================================== *)
(* What is known of a unit                                                  *)
(* ======================================================================== *)

(* Every constant that is known has a value. *)
Definition all_have_values (known : list (symbol * static_const)) : Prop :=
  Forall (fun binding => eval_static_const [] (snd binding) <> None) known.

Lemma known_of_binding_have_values (binding : static_binding) :
  all_have_values (known_of_binding binding).
Proof.
  destruct binding as [id c | id | s const]; simpl; try constructor.
  destruct (closed_static_const const); [ | constructor ].
  destruct (eval_static_const [] const) eqn:Hconst; constructor;
    [ simpl; congruence | constructor ].
Qed.

Fixpoint known_bindings_have_values (e : expr) {struct e} :
  all_have_values (known_bindings e)
with known_bindings_of_handler_have_values (h : continuation_handler)
    {struct h} :
  all_have_values (known_bindings_of_handler h).
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try constructor.
    + apply known_bindings_have_values.
    + apply Forall_app. split; [ | apply known_bindings_have_values ].
      induction bindings as [ | binding bindings IH]; simpl;
        [ constructor | ].
      apply Forall_app. split; [ apply known_of_binding_have_values | ].
      exact IH.
    + apply Forall_app.
      split;
        [ apply known_bindings_of_handler_have_values
        | apply known_bindings_have_values ].
    + apply Forall_app. split; [ | apply known_bindings_have_values ].
      induction handlers as [ | [k h] handlers IH]; simpl; [ constructor | ].
      apply Forall_app.
      split; [ apply known_bindings_of_handler_have_values | exact IH ].
  - destruct h as [params metadata handler]. simpl.
    apply known_bindings_have_values.
Qed.

(* The fields of a static block, with [unknown_field] for those that are
   variables, have values with no variable in scope; and those values
   are the fields that the block is made with wherever they are not
   [unknown_field]. *)
Definition masked (fields : list simple) : list simple :=
  map (fun field => if closed_simple field then field else Const unknown_field)
    fields.

Lemma masked_fields_have_values (fields : list simple) :
  ∃ vs : list value, eval_simples [] (masked fields) = Some vs.
Proof.
  unfold masked.
  induction fields as [ | field fields [vs IH]]; simpl; [ eauto | ].
  rewrite IH.
  destruct field as [x | s | c]; simpl; eauto.
Qed.

Lemma masked_fields_known (vars : env) (fields : list simple) :
  ∀ known actual : list value,
    eval_simples [] (masked fields) = Some known ->
    eval_simples vars fields = Some actual ->
    Forall2
      (fun known actual => known = Value_const unknown_field ∨ actual = known)
      known actual.
Proof.
  unfold masked.
  induction fields as [ | field fields IH]; simpl; intros known actual.
  - intros [= <-] [= <-]. constructor.
  - destruct
      (eval_simples []
         (map
            (fun field =>
               if closed_simple field then field else Const unknown_field)
            fields))
      as [known_rest | ] eqn:Hrest;
      [ | destruct (closed_simple field);
          [ destruct (eval_simple [] field) | ]; discriminate ].
    destruct (eval_simple vars field) as [v | ] eqn:Hv; simpl;
      [ | discriminate ].
    destruct (eval_simples vars fields) as [actual_rest | ]; simpl;
      [ | destruct (closed_simple field);
          [ destruct (eval_simple [] field) | ]; discriminate ].
    destruct (closed_simple field) eqn:Hclosed.
    + rewrite <- (eval_closed_simple vars field Hclosed), Hv. simpl.
      intros [= <-] [= <-].
      constructor; [ right; reflexivity | exact (IH _ _ eq_refl eq_refl) ].
    + simpl. intros [= <-] [= <-].
      constructor; [ left; reflexivity | exact (IH _ _ eq_refl eq_refl) ].
Qed.

Lemma set_of_binding_have_values (binding : static_binding) :
  all_have_values (set_of_binding binding).
Proof.
  destruct binding as [id c | id | s const]; try constructor.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements].
  - unfold all_have_values, set_of_binding.
    constructor; [ discriminate | constructor ].
  - unfold all_have_values, set_of_binding. cbv zeta.
    destruct
      (eval_static_const [] (Static_const_closure_of_set fslot syms decls []))
      eqn:Hconst;
      constructor; [ | constructor ].
    cbn [snd]. rewrite Hconst. discriminate.
  - unfold all_have_values, set_of_binding.
    constructor; [ | constructor ].
    cbn [snd eval_static_const].
    destruct (masked_fields_have_values fields) as [vs Hvs].
    unfold masked in Hvs. rewrite Hvs. discriminate.
  - constructor.
  - constructor.
  - constructor.
Qed.

Fixpoint known_sets_have_values (e : expr) {struct e} :
  all_have_values (known_sets e)
with known_sets_of_handler_have_values (h : continuation_handler)
    {struct h} :
  all_have_values (known_sets_of_handler h).
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try constructor.
    + apply known_sets_have_values.
    + apply Forall_app. split; [ | apply known_sets_have_values ].
      induction bindings as [ | binding bindings IH]; simpl;
        [ constructor | ].
      apply Forall_app. split; [ apply set_of_binding_have_values | ].
      exact IH.
    + apply Forall_app.
      split;
        [ apply known_sets_of_handler_have_values
        | apply known_sets_have_values ].
    + apply Forall_app. split; [ | apply known_sets_have_values ].
      induction handlers as [ | [k h] handlers IH]; simpl; [ constructor | ].
      apply Forall_app.
      split; [ apply known_sets_of_handler_have_values | exact IH ].
  - destruct h as [params metadata handler]. simpl.
    apply known_sets_have_values.
Qed.

(* What [known_statics] binds a symbol to is the value of the constant
   that is known of it. *)
Lemma lookup_known_statics (known : list (symbol * static_const))
    (s : symbol) :
  all_have_values known ->
  lookup s (static_values (known_statics known))
  = match lookup s known with
    | Some const => eval_static_const [] const
    | None => None
    end.
Proof.
  unfold known_statics. simpl.
  induction 1 as [ | [s' const] known Hconst _ IH]; simpl; [ reflexivity | ].
  unfold known_value at 1. simpl in Hconst |- *.
  destruct (eval_static_const [] const) as [allocated | ] eqn:Heval;
    [ | contradiction Hconst; reflexivity ].
  simpl. destruct (Pos.eqb s s'); [ symmetry; exact Heval | exact IH ].
Qed.

(* ======================================================================== *)
(* With the run of the machine                                              *)
(* ======================================================================== *)

(* The machine runs a [Let] with the heap ([exec_named]); the three
   folds are stated of the value alone ([eval_named]). A defining
   expression that a fold changes is a projection, or a primitive that
   is not of the heap and has no variable among its arguments. No
   location is among the values of such arguments, so the machine runs
   it as [eval_named] says and leaves the heap and the record as they
   came; and what a fold puts for it is a symbol or a constant, which
   the machine runs without the heap too. *)
Definition closed_simple (a : simple) : bool :=
  match a with
  | Var _ => false
  | Symbol _ | Const _ => true
  end.

Definition run_without_heap (d : named) : bool :=
  match d with
  | Prim (Heap _) _ => false
  | Prim (Unary _ | Binary _ | Variadic _ | Region_primitive _) args =>
    forallb closed_simple args
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    true
  end.

Lemma closed_simples_no_location (vars : env) (args : list simple) :
  forallb closed_simple args = true ->
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    ∀ l : location, Value_location l ∉ vs.
Proof.
  induction args as [ | a args IH]; simpl; intros H vs.
  - intros [= <-] l [].
  - apply andb_true_iff in H as [Ha Hargs].
    destruct (eval_simple vars a) as [v | ] eqn:Hv; simpl; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; simpl; [ | discriminate ].
    intros [= <-] l [Hhead | Hrest];
      [ | exact (IH Hargs rest eq_refl l Hrest) ].
    subst v. destruct a as [x | sym | c]; simpl in Hv, Ha;
      discriminate.
Qed.

Lemma exec_named_without_heap (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  run_without_heap d = true ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  eval_named statics vars d = Some v ∧ hp' = hp ∧ rd' = rd.
Proof.
  intros Hd Hexec.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (rewrite exec_named_not_prim in Hexec by (intros ? ?; discriminate);
         destruct (eval_named _ _ _) as [w | ]; [ | discriminate ];
         injection Hexec as <- <- <-; repeat split; reflexivity).
  assert (Hnot : ∀ q : heap_primitive, p <> Heap q)
    by (intros q ->; discriminate Hd).
  assert (Hargs : forallb closed_simple args = true)
    by (destruct p; first [ exact Hd | discriminate Hd ]).
  pose proof Hexec as Hvs.
  apply exec_named_prim_some in Hvs as (vs & Hvs & _ & _).
  rewrite
    (exec_named_no_location statics hp rd vars p args vs Hnot Hvs
       (closed_simples_no_location vars args Hargs vs Hvs))
    in Hexec.
  destruct (eval_named statics vars (Prim p args)) as [w | ];
    [ | discriminate ].
  injection Hexec as <- <- <-. repeat split; reflexivity.
Qed.

(* So one fold, from what is proved of it for [eval_named]. *)
Lemma folded_exec (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d d' : named) (v : value) :
  (eval_named statics vars d = Some v ->
   eval_named statics vars d' = Some v) ->
  d' = d ∨ (run_without_heap d = true ∧ ∃ a : simple, d' = Simple a) ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars d' = Some (v, hp', rd').
Proof.
  intros Hsound [-> | (Hd & a & ->)] Hexec; [ exact Hexec | ].
  destruct (exec_named_without_heap statics hp hp' rd rd' vars d v Hd Hexec)
    as (Heval & -> & ->).
  rewrite exec_named_not_prim by (intros ? ?; discriminate).
  rewrite (Hsound Heval). reflexivity.
Qed.

(* What each fold does to a defining expression: nothing, or it puts a
   symbol or a constant for one that the machine runs without the
   heap. *)
Lemma fold_known_named_cases (statics : static_env) (d : named) :
  fold_known_named statics d = d
  ∨ (run_without_heap d = true
     ∧ ∃ a : simple, fold_known_named statics d = Simple a).
Proof.
  unfold fold_known_named.
  destruct (named_known statics d) eqn:Hknown; [ | left; reflexivity ].
  assert (Hd : run_without_heap d = true).
  { destruct d as [s | p args | code slots | fslot decls slots | slot closure
      | pfrom slot closure | move_from move_to closure];
      simpl in Hknown |- *; try reflexivity; try discriminate Hknown.
    apply andb_true_iff in Hknown as [Hp Hargs].
    apply andb_true_iff in Hp as [Hp _].
    assert (Hclosed : forallb closed_simple args = true).
    { apply forallb_forall. intros a Ha.
      pose proof (proj1 (forallb_forall _ _) Hargs a Ha) as Hknown.
      destruct a; [ discriminate Hknown | reflexivity | reflexivity ]. }
    destruct p; first [ exact Hclosed | discriminate Hp ]. }
  destruct (eval_named statics [] d) as [[ ] | ];
    first
      [ left; reflexivity
      | right; split; [ exact Hd | eexists; reflexivity ] ].
Qed.

Lemma fold_function_slot_cases (sets : static_env) (d : named) :
  fold_function_slot sets d = d
  ∨ (run_without_heap d = true
     ∧ ∃ a : simple, fold_function_slot sets d = Simple a).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (left; reflexivity).
  destruct closure as [x | sym | c]; try (left; reflexivity).
  unfold fold_function_slot.
  destruct (eval_named sets [] _) as [[ ] | ];
    first
      [ left; reflexivity
      | right; split; [ reflexivity | eexists; reflexivity ] ].
Qed.

Lemma fold_block_load_cases (sets : static_env) (d : named) :
  fold_block_load sets d = d
  ∨ (run_without_heap d = true
     ∧ ∃ a : simple, fold_block_load sets d = Simple a).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (left; reflexivity).
  destruct p as [p | p | p | p | rp]; try (left; reflexivity).
  destruct p; try (left; reflexivity).
  destruct args as [ | [x | s | c] [ | arg args]]; try (left; reflexivity);
    try (destruct mut; left; reflexivity).
  destruct mut; [ left; reflexivity | | ];
    unfold fold_block_load;
    (destruct (eval_named sets [] _) as [[ ] | ];
       first
         [ left; reflexivity
         | right; split; [ reflexivity | eexists; reflexivity ] ]).
Qed.

(* ======================================================================== *)
(* The run of a unit                                                        *)
(* ======================================================================== *)

Section Unit.

Variable known : list (symbol * static_const).
Hypothesis Hknown : all_have_values known.
Variable sets : list (symbol * static_const).
Hypothesis Hsets : all_have_values sets.

Let S : static_env := known_statics known.
Let T : static_env := known_statics sets.

(* What the run has bound of the known symbols is what is known. *)
Definition agrees_known (run_statics : static_env) : Prop :=
  ∀ (s : symbol) (allocated allocated' : statically_allocated_value),
    lookup s (static_values S) = Some allocated ->
    lookup s (static_values run_statics) = Some allocated' ->
    allocated' = allocated.

(* What it has bound of the symbols of closures of sets are closures at
   the slots and of the sets that are known. *)
Definition agrees_sets (run_statics : static_env) : Prop :=
  ∀ (s : symbol) (allocated allocated' : statically_allocated_value),
    lookup s (static_values T) = Some allocated ->
    lookup s (static_values run_statics) = Some allocated' ->
    same_set allocated allocated'.

Definition agrees (run_statics : static_env) : Prop :=
  agrees_known run_statics ∧ agrees_sets run_statics.

Lemma eval_static_bindings_agrees_sets (vars : env)
    (bindings : list static_binding) :
  forallb (binding_bound_as known sets) bindings = true ->
  ∀ statics statics' : static_env,
    agrees_sets statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    agrees_sets statics'.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH]; simpl;
    intros Hbound statics statics' Hagrees.
  - intros [= <-]. exact Hagrees.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    apply IH; [ exact Hbound | ]. exact Hagrees.
  - apply IH; assumption.
  - apply andb_true_iff in Hbound as [Hsym Hbound].
    apply andb_true_iff in Hsym as [_ Hsym].
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [made | ] eqn:Hmade; simpl;
      [ | discriminate ].
    apply IH; [ exact Hbound | ].
    intros s allocated allocated' Hs. simpl.
    destruct (Pos.eqb_spec s sym) as [-> | _];
      [ | exact (Hagrees s allocated allocated' Hs) ].
    intros [= <-]. unfold T in Hs.
    rewrite (lookup_known_statics sets sym Hsets) in Hs.
    destruct (lookup sym sets) as [expected | ]; [ | discriminate ].
    destruct const
      as [code slots | fslot syms decls slots | tag fields | contents
         | bytes | elements];
      try discriminate.
    { cbn -[static_const_eq_dec] in Hsym.
      destruct
        (static_const_eq_dec (Static_const_set_of_closures code []) expected)
        as [<- | ];
        [ | discriminate ].
      simpl in Hs. injection Hs as <-. simpl in Hmade.
      destruct (eval_value_slots vars slots) as [vs | ]; simpl in Hmade;
        [ | discriminate ].
      injection Hmade as <-. reflexivity. }
    { simpl in Hmade.
      destruct (lookup fslot decls) as [code | ] eqn:Hcode;
        cbn -[static_const_eq_dec] in Hsym, Hmade;
        [ | discriminate ].
      destruct
        (static_const_eq_dec (Static_const_closure_of_set fslot syms decls [])
           expected)
        as [<- | ];
        [ | discriminate ].
      simpl in Hs. rewrite Hcode in Hs. simpl in Hs. injection Hs as <-.
      destruct (eval_value_slots vars slots) as [vs | ]; simpl in Hmade;
        [ | discriminate ].
      injection Hmade as <-. simpl. auto. }
    cbn -[static_const_eq_dec] in Hsym.
    fold (masked fields) in Hsym.
    destruct
      (static_const_eq_dec (Static_const_block tag (masked fields)) expected)
      as [<- | ];
      [ | discriminate ].
    cbn [eval_static_const] in Hs, Hmade.
    destruct (eval_simples [] (masked fields)) as [known_fields | ] eqn:Hkf;
      simpl in Hs; [ | discriminate ].
    injection Hs as <-.
    destruct (eval_simples vars fields) as [vs | ] eqn:Hvs; simpl in Hmade;
      [ | discriminate ].
    injection Hmade as <-.
    split; [ reflexivity | ].
    exact (masked_fields_known vars fields known_fields vs Hkf Hvs).
Qed.

Lemma eval_static_bindings_agrees_known (vars : env)
    (bindings : list static_binding) :
  forallb (binding_bound_as known sets) bindings = true ->
  ∀ statics statics' : static_env,
    agrees_known statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    agrees_known statics'.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH]; simpl;
    intros Hbound statics statics' Hagrees.
  - intros [= <-]. exact Hagrees.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    apply IH; [ exact Hbound | ]. exact Hagrees.
  - apply IH; assumption.
  - apply andb_true_iff in Hbound as [Hsym Hbound].
    apply andb_true_iff in Hsym as [Hsym _].
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [made | ] eqn:Hmade; simpl;
      [ | discriminate ].
    apply IH; [ exact Hbound | ].
    intros s allocated allocated' Hs. simpl.
    destruct (Pos.eqb_spec s sym) as [-> | _];
      [ | exact (Hagrees s allocated allocated' Hs) ].
    intros [= <-]. unfold S in Hs.
    rewrite (lookup_known_statics known sym Hknown) in Hs.
    destruct (lookup sym known) as [expected | ]; [ | discriminate ].
    apply andb_true_iff in Hsym as [Hclosed Heq].
    destruct (static_const_eq_dec const expected) as [<- | ];
      [ | discriminate ].
    rewrite (eval_closed_static_const vars const Hclosed) in Hmade.
    congruence.
Qed.

Lemma eval_static_bindings_agrees (vars : env)
    (bindings : list static_binding) :
  forallb (binding_bound_as known sets) bindings = true ->
  ∀ statics statics' : static_env,
    agrees statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    agrees statics'.
Proof.
  intros Hbound statics statics' [Hknown_agrees Hsets_agrees] Heval.
  split;
    [ exact (eval_static_bindings_agrees_known vars bindings Hbound statics
               statics' Hknown_agrees Heval)
    | exact (eval_static_bindings_agrees_sets vars bindings Hbound statics
               statics' Hsets_agrees Heval) ].
Qed.

(* The static environment of the second state: that of the first with
   its code folded. *)
Definition fold_statics (statics : static_env) : static_env :=
  Static_env
    (map (fun binding => (fst binding, fold_known_code S T (snd binding)))
       (static_code statics))
    (static_values statics).

Lemma lookup_fold_code (id : code_id) (statics : static_env) :
  lookup id (static_code (fold_statics statics))
  = option_map (fold_known_code S T) (lookup id (static_code statics)).
Proof. unfold fold_statics. apply lookup_map_snd. Qed.

(* The value of a defining expression does not depend on the code. *)
Lemma eval_named_fold_statics (statics : static_env) (vars : env)
    (d : named) :
  eval_named (fold_statics statics) vars d = eval_named statics vars d.
Proof.
  destruct d; simpl; try reflexivity.
  all:
    destruct (eval_simples vars args) as [vs | ]; simpl; [ | reflexivity ];
    apply eval_prim_static_values; intros s _; reflexivity.
Qed.

(* Nor does what it does on the heap. *)
Lemma exec_named_fold_statics (statics : static_env) (hp : heap)
    (rd : read_record) (vars : env) (d : named) :
  exec_named (fold_statics statics) hp rd vars d
  = exec_named statics hp rd vars d.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (rewrite !exec_named_not_prim by (intros ? ?; discriminate);
         rewrite eval_named_fold_statics; reflexivity).
  apply exec_named_static_values. intros vs s _ _. reflexivity.
Qed.

(* The defining expression of a [Let] with the three folds: what the
   machine runs it to, it runs the folded one to, under the static
   environment with its code folded. *)
Lemma fold_named_exec (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  agrees statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named (fold_statics statics) hp rd vars
    (fold_block_load T (fold_function_slot T (fold_known_named S d)))
  = Some (v, hp', rd').
Proof.
  intros Hagrees Hexec. rewrite exec_named_fold_statics.
  apply (folded_exec statics hp hp' rd rd' vars
           (fold_function_slot T (fold_known_named S d)));
    [ exact (fold_block_load_sound_here T statics (proj2 Hagrees) vars _ v)
    | apply fold_block_load_cases | ].
  apply (folded_exec statics hp hp' rd rd' vars (fold_known_named S d));
    [ exact (fold_function_slot_sound_here T statics (proj2 Hagrees) vars _
               v)
    | apply fold_function_slot_cases | ].
  apply (folded_exec statics hp hp' rd rd' vars d);
    [ exact (fold_known_named_sound_here S statics (proj1 Hagrees) vars d v)
    | apply fold_known_named_cases | exact Hexec ].
Qed.

Lemma eval_static_bindings_fold (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    eval_static_bindings vars bindings statics = Some statics' ->
    eval_static_bindings vars (map (fold_known_binding S T) bindings)
      (fold_statics statics)
    = Some (fold_statics statics').
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros [= <-]. reflexivity.
  - rewrite lookup_map_snd.
    destruct (lookup id (static_code statics)); simpl; [ discriminate | ].
    intros H. exact (IH _ _ H).
  - apply IH.
  - destruct (lookup sym (static_values statics)); simpl; [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | discriminate ].
    intros H. exact (IH _ _ H).
Qed.

(* The rewrite keeps whether a term binds something static. *)
Fixpoint fold_known_binds_statics (e : expr) {struct e} :
  binds_statics (fold_known S T e) = binds_statics e
with fold_known_handler_binds_statics (h : continuation_handler)
    {struct h} :
  handler_binds_statics (fold_known_handler S T h) = handler_binds_statics h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply fold_known_binds_statics.
    + rewrite fold_known_binds_statics. f_equal.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite IH. destruct binding; reflexivity.
    + rewrite fold_known_handler_binds_statics, fold_known_binds_statics.
      reflexivity.
    + rewrite fold_known_binds_statics. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite fold_known_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply fold_known_binds_statics.
Qed.

Lemma fold_known_handlers_bind_statics
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, fold_known_handler S T h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite fold_known_handler_binds_statics, IH. reflexivity.
Qed.

Lemma fold_known_code_parts (c : code) :
  code_params (fold_known_code S T c) = code_params c
  ∧ code_my_closure (fold_known_code S T c) = code_my_closure c
  ∧ code_return_continuation (fold_known_code S T c)
    = code_return_continuation c
  ∧ code_exn_continuation (fold_known_code S T c) = code_exn_continuation c
  ∧ code_params_arity (fold_known_code S T c) = code_params_arity c
  ∧ code_result_arity (fold_known_code S T c) = code_result_arity c
  ∧ code_body (fold_known_code S T c) = fold_known S T (code_body c).
Proof. destruct c. simpl. auto 8. Qed.

(* A call of the second run has the shape of the call of the first,
   with the same values, of the code folded. *)
Definition fold_shape (shape : call_shape) : call_shape :=
  match shape with
  | Full_call closure c vs args_arity =>
    Full_call closure (fold_known_code S T c) vs args_arity
  | Over_call closure c vs remaining args_arity remaining_arity =>
    Over_call closure (fold_known_code S T c) vs remaining args_arity
      remaining_arity
  | Partial_call partial => Partial_call partial
  end.

Lemma shape_of_closure_call_fold (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some shape ->
  shape_of_closure_call (fold_statics statics) closure kind vs args_arity
    return_arity
  = Some (fold_shape shape).
Proof.
  unfold shape_of_closure_call.
  change (closure_of (fold_statics statics) closure)
    with (closure_of statics closure).
  destruct (closure_of statics closure) as [[called slots] | ];
    [ | discriminate ].
  assert (Hfull :
    known_arity_full (fold_statics statics) called args_arity
    = known_arity_full statics called args_arity).
  { apply (known_arity_full_mapped (fold_known_code S T));
      [ apply lookup_fold_code | ].
    intros c _. destruct c. split; reflexivity. }
  rewrite Hfull.
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect_unknown_arity => true
     | Indirect_known_arity code_ids =>
       code_ids_allow code_ids called
       && known_arity_full statics called args_arity
     end);
    [ | discriminate ].
  rewrite lookup_fold_code.
  destruct (lookup called (static_code statics)) as [c | ]; [ | discriminate ].
  simpl option_map. cbv iota zeta.
  destruct (fold_known_code_parts c)
    as (_ & _ & _ & _ & Hparams_arity & Hresult_arity & _).
  assert (Htupled : code_is_tupled (fold_known_code S T c) = code_is_tupled c)
    by (destruct c; reflexivity).
  destruct (arguments_for_code statics c vs args_arity)
    as [[vs' args_arity'] | ] eqn:Hargs; [ | discriminate ].
  rewrite (arguments_for_code_mono statics (fold_statics statics) c
             (fold_known_code S T c) vs args_arity _ Htupled Hparams_arity
             (fun tuple rest p v _ Hp =>
                eq_trans
                  (eq_sym
                     (eval_prim_same_static_values statics
                        (fold_statics statics) p [tuple] eq_refl))
                  Hp)
             Hargs).
  simpl. unfold shape_of_code_call. cbv zeta.
  rewrite Hparams_arity, Hresult_arity, Htupled.
  destruct (Nat.eqb (length args_arity') (length (code_params_arity c)));
    [ intros [= <-]; reflexivity | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs') (unarize args_arity')
        && partial_or_over_application args_arity' return_arity
             (code_params_arity c) (code_result_arity c)));
    [ discriminate | ].
  destruct (Nat.ltb (length (code_params_arity c)) (length args_arity'));
    [ intros [= <-]; reflexivity | ].
  destruct (code_is_tupled c); [ discriminate | ].
  intros [= <-]. reflexivity.
Qed.

Lemma shape_of_call_fold (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  shape_of_call statics callee kind vs args_arity return_arity = Some shape ->
  shape_of_call (fold_statics statics) callee kind vs args_arity return_arity
  = Some (fold_shape shape).
Proof.
  unfold shape_of_call.
  destruct kind as [named | | code_ids];
    [ apply shape_of_closure_call_fold | | apply shape_of_closure_call_fold ].
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ];
    [ | apply shape_of_closure_call_fold ].
  destruct args_arity as [ | component args_arity]; [ discriminate | ].
  destruct (kinds_eqb _ _); [ | discriminate ].
  apply shape_of_closure_call_fold.
Qed.

(* The terms of a state bind the known symbols as is known. *)
Definition bound_as_inv : state -> Prop :=
  terms_inv (bound_as known sets) (handler_bound_as known sets)
    (code_bound_as known sets).

Lemma bound_as_inv_step (s s' : state) :
  bound_as_inv s -> step s = Step s' -> bound_as_inv s'.
Proof.
  apply terms_inv_step.
  - intros x xkind d body H. exact H.
  - intros bindings body H. simpl in H.
    apply andb_true_iff in H as [H Hbody].
    apply andb_true_iff in H as [_ Hcode].
    split; [ exact Hbody | ]. intros id c Hin.
    exact (proj1 (forallb_forall _ _) Hcode (Bind_code id c) Hin).
  - intros k h body H. simpl in H. apply andb_true_iff in H. tauto.
  - intros handlers body H. simpl in H. apply andb_true_iff in H. tauto.
  - intros ac. reflexivity.
  - intros params metadata handler H. exact H.
  - intros [params my_closure ret exn metadata body] H. exact H.
Qed.

(* The second state. The unit may be followed by something, and what
   may follow a unit is not folded in the second run: it is the same
   term in the two. So the second state is given by a relation and not
   as a function of the first: each term that the first state keeps,
   its expression and the handlers of its continuations, is folded in
   the second or, if it binds nothing static, left as it is. The code
   is folded. A term that is left as it is binds no code, so the static
   environment of the second state is still that of the first with its
   code folded. *)
Definition expr_kept (e e' : expr) : Prop :=
  e' = fold_known S T e ∨ (e' = e ∧ binds_no_static e = true).

Definition handler_kept (h h' : continuation_handler) : Prop :=
  h' = fold_known_handler S T h
  ∨ (h' = h ∧ handler_binds_no_static h = true).

Definition handlers_kept
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  handlers' = map (fun '(k, h) => (k, fold_known_handler S T h)) handlers
  ∨ (handlers' = handlers
     ∧ forallb (fun '(_, h) => handler_binds_no_static h) handlers = true).

#[local] Set Warnings "-register-all".
Inductive cont_kept : cont_value -> cont_value -> Prop :=
  | Closure_kept (h h' : continuation_handler) (vars : env)
      (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    handler_kept h h' ->
    handlers_kept recursive recursive' ->
    Forall2
      (fun kc kc' => fst kc = fst kc' ∧ cont_kept (snd kc) (snd kc'))
      conts conts' ->
    cont_kept (Closure h vars conts recursive)
      (Closure h' vars conts' recursive')
  | Outer_kept (k : continuation) : cont_kept (Outer k) (Outer k)
  | Extra_args_kept (c c' : cont_value) (extra : list value) :
    cont_kept c c' -> cont_kept (Extra_args c extra) (Extra_args c' extra)
  | Over_application_kept (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_kept return_to return_to' ->
    cont_kept exn_to exn_to' ->
    cont_kept
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Definition conts_kept (conts conts' : cont_env) : Prop :=
  Forall2 (fun kc kc' => fst kc = fst kc' ∧ cont_kept (snd kc) (snd kc'))
    conts conts'.

Inductive state_kept : state -> state -> Prop :=
  | State_kept (e e' : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    expr_kept e e' ->
    conts_kept conts conts' ->
    state_kept (State e vars conts statics hp rd)
      (State e' vars conts' (fold_statics statics) hp rd).

(* What the second run does where the first does something: a step to
   a state so related, or the same exit. *)
Definition result_kept (result result' : step_result) : Prop :=
  match result with
  | Step t => ∃ t' : state, result' = Step t' ∧ state_kept t t'
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Undefined => True
  end.

Lemma resolve_kept (k : continuation) (conts conts' : cont_env) :
  conts_kept conts conts' -> cont_kept (resolve k conts) (resolve k conts').
Proof.
  unfold resolve.
  induction 1 as [ | [k1 c] [k1' c'] conts conts' [Hk Hc] _ IH]; simpl;
    [ apply Outer_kept | ].
  simpl in Hk. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

Lemma with_extra_args_kept (c c' : cont_value) (extra : list value) :
  cont_kept c c' ->
  cont_kept (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  destruct extra; intros Hc; [ exact Hc | apply Extra_args_kept; exact Hc ].
Qed.

Lemma rec_closures_kept
    (handlers handlers' : list (continuation * continuation_handler))
    (vars : env) (conts conts' : cont_env) :
  handlers_kept handlers handlers' ->
  conts_kept conts conts' ->
  conts_kept (rec_closures handlers vars conts)
    (rec_closures handlers' vars conts').
Proof.
  intros Hhandlers Hconts. unfold rec_closures.
  destruct Hhandlers as [-> | [-> Hno]].
  - assert (Hall :
      ∀ group : list (continuation * continuation_handler),
        conts_kept
          (map (fun '(k, h) => (k, Closure h vars conts handlers)) group)
          (map
             (fun '(k, h) =>
                (k,
                 Closure h vars conts'
                   (map (fun '(k, h) => (k, fold_known_handler S T h))
                      handlers)))
             (map (fun '(k, h) => (k, fold_known_handler S T h)) group))).
    { induction group as [ | [k h] group IH]; simpl; constructor;
        [ | exact IH ].
      split; [ reflexivity | ].
      apply Closure_kept;
        [ left; reflexivity | left; reflexivity | exact Hconts ]. }
    exact (Hall handlers).
  - assert (Hall :
      ∀ group : list (continuation * continuation_handler),
        forallb (fun '(_, h) => handler_binds_no_static h) group = true ->
        conts_kept
          (map (fun '(k, h) => (k, Closure h vars conts handlers)) group)
          (map (fun '(k, h) => (k, Closure h vars conts' handlers)) group)).
    { induction group as [ | [k h] group IH]; simpl; [ constructor | ].
      intros H. apply andb_true_iff in H as [Hh Hrest].
      constructor; [ | exact (IH Hrest) ].
      split; [ reflexivity | ].
      apply Closure_kept;
        [ right; split; [ reflexivity | exact Hh ]
        | right; split; [ reflexivity | exact Hno ] | exact Hconts ]. }
    exact (Hall handlers Hno).
Qed.

(* A full call. *)
Lemma call_code_kept (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  cont_kept return_to return_to' ->
  cont_kept exn_to exn_to' ->
  result_kept
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure (fold_known_code S T c) vs args_arity return_arity
       return_to' exn_to' (fold_statics statics) hp rd).
Proof.
  intros Hreturn Hexn. unfold result_kept, call_code.
  destruct (fold_known_code_parts c)
    as (Hparams & Hmy_closure & Hret & Hexn_c & Hparams_arity & Hresult_arity
        & Hbody).
  rewrite Hparams, Hmy_closure, Hret, Hexn_c, Hbody, fold_known_binds_statics.
  destruct (binds_statics (code_body c)); [ exact I | ].
  rewrite Hparams_arity, Hresult_arity.
  match goal with
  | |- context [negb ?agrees] => destruct (negb agrees)
  end;
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ];
    [ | exact I ].
  eexists. split; [ reflexivity | ].
  apply State_kept; [ left; reflexivity | ].
  constructor; [ split; [ reflexivity | exact Hreturn ] | ].
  constructor; [ split; [ reflexivity | exact Hexn ] | constructor ].
Qed.

Lemma run_call_kept (return_partial return_partial' : value -> step_result)
    (shape : call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  cont_kept return_to return_to' ->
  cont_kept exn_to exn_to' ->
  (∀ partial : value,
    result_kept (return_partial partial) (return_partial' partial)) ->
  result_kept
    (run_call return_partial (Some shape) return_arity return_to exn_to
       statics hp rd)
    (run_call return_partial' (Some (fold_shape shape)) return_arity
       return_to' exn_to' (fold_statics statics) hp rd).
Proof.
  intros Hreturn Hexn Hpartial.
  destruct shape
    as [closure c vs args_arity
       | closure c vs remaining args_arity remaining_arity | partial];
    simpl.
  - apply call_code_kept; assumption.
  - destruct (fold_known_code_parts c) as (_ & _ & _ & _ & _ & -> & _).
    apply call_code_kept;
      [ apply Over_application_kept; assumption | exact Hexn ].
  - apply Hpartial.
Qed.

Lemma call_function_kept_from (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  cont_kept return_to return_to' ->
  cont_kept exn_to exn_to' ->
  (∀ partial : value,
    result_kept (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] (fold_statics statics) hp rd)) ->
  result_kept
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       (fold_statics statics) hp rd).
Proof.
  intros Hreturn Hexn Hpartial. unfold call_function.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | exact I ].
  rewrite (shape_of_call_fold statics callee kind vs args_arity return_arity
             shape Hshape).
  apply run_call_kept; assumption.
Qed.

Lemma apply_cont_value_kept (c c' : cont_value) :
  cont_kept c c' ->
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    result_kept (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs (fold_statics statics) hp rd).
Proof.
  induction 1
    as [ h h' def_vars def_conts def_conts' recursive recursive' Hh Hrec
           Hconts
       | k | c c' extra Hc IH
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hexn _ ];
    intros vs statics hp rd.
  - assert (Hparts :
      ∃ (params : list variable) (metadata : handler_metadata)
        (body body' : expr),
        h = Handler params metadata body
        ∧ h' = Handler params metadata body' ∧ expr_kept body body').
    { destruct h as [params metadata body]. destruct Hh as [-> | [-> Hno]].
      - exists params, metadata, body, (fold_known S T body).
        split; [ reflexivity | ]. split; [ reflexivity | ].
        left. reflexivity.
      - exists params, metadata, body, body.
        split; [ reflexivity | ]. split; [ reflexivity | ].
        right. split; [ reflexivity | exact Hno ]. }
    destruct Hparts as (params & metadata & body & body' & -> & -> & Hbody).
    unfold result_kept. simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      [ | exact I ].
    eexists. split; [ reflexivity | ].
    apply State_kept; [ exact Hbody | ].
    apply Forall2_app;
      [ exact (rec_closures_kept recursive recursive' def_vars def_conts
                 def_conts' Hrec Hconts)
      | exact Hconts ].
  - reflexivity.
  - simpl. apply IH.
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    apply call_function_kept_from; [ exact Hreturn | exact Hexn | ].
    intros partial. apply IHreturn.
Qed.

Lemma call_function_kept (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  cont_kept return_to return_to' ->
  cont_kept exn_to exn_to' ->
  result_kept
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       (fold_statics statics) hp rd).
Proof.
  intros Hreturn Hexn.
  apply call_function_kept_from; [ exact Hreturn | exact Hexn | ].
  intros partial. apply apply_cont_value_kept. exact Hreturn.
Qed.

(* A step of a state that binds the known symbols as is known, and has
   bound them so, is answered by a step of a state so related, to a
   state so related. *)
Lemma kept_step_components (e e' : expr) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  expr_kept e e' ->
  conts_kept conts conts' ->
  bound_as known sets e = true ->
  agrees statics ->
  match step (State e vars conts statics hp rd) with
  | Step t =>
    (∃ t' : state,
      step (State e' vars conts' (fold_statics statics) hp rd) = Step t'
      ∧ state_kept t t')
    ∧ agrees (state_statics t)
  | Exit k vs hp' rd' =>
    step (State e' vars conts' (fold_statics statics) hp rd) = Exit k vs hp' rd'
  | Undefined => True
  end.
Proof.
  intros He Hconts Hbound Hagrees.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - (* Let *)
    rewrite step_let_exec.
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hv;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) xkind) eqn:Hkind; [ | exact I ].
    split; [ | exact Hagrees ].
    destruct He as [-> | [-> Hno]].
    + simpl fold_known.
      rewrite step_let_exec,
        (fold_named_exec statics hp hp' rd rd' vars d v Hagrees Hv), Hkind.
      eexists. split; [ reflexivity | ].
      apply State_kept; [ left; reflexivity | exact Hconts ].
    + rewrite step_let_exec, exec_named_fold_statics, Hv, Hkind.
      eexists. split; [ reflexivity | ].
      apply State_kept; [ right; split; [ reflexivity | exact Hno ] | ].
      exact Hconts.
  - (* Let_static *)
    rewrite step_let_static.
    simpl in Hbound. apply andb_true_iff in Hbound as [Hbound _].
    apply andb_true_iff in Hbound as [Hbindings _].
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | exact I ].
    split;
      [ | exact (eval_static_bindings_agrees vars bindings Hbindings statics
                   statics' Hagrees Heval) ].
    destruct He as [-> | [_ Hno]]; [ | discriminate Hno ].
    simpl fold_known. rewrite step_let_static.
    rewrite (eval_static_bindings_fold vars bindings statics statics' Heval).
    eexists. split; [ reflexivity | ].
    apply State_kept; [ left; reflexivity | exact Hconts ].
  - (* Let_cont_non_recursive *)
    rewrite step_let_cont_non_recursive. split; [ | exact Hagrees ].
    destruct He as [-> | [-> Hno]].
    + simpl fold_known. rewrite step_let_cont_non_recursive.
      eexists. split; [ reflexivity | ].
      apply State_kept; [ left; reflexivity | ].
      constructor; [ split; [ reflexivity | ] | exact Hconts ].
      apply Closure_kept;
        [ left; reflexivity | left; reflexivity | exact Hconts ].
    + rewrite step_let_cont_non_recursive.
      simpl in Hno. apply andb_true_iff in Hno as [Hno_h Hno_b].
      eexists. split; [ reflexivity | ].
      apply State_kept; [ right; split; [ reflexivity | exact Hno_b ] | ].
      constructor; [ split; [ reflexivity | ] | exact Hconts ].
      apply Closure_kept;
        [ right; split; [ reflexivity | exact Hno_h ] | left; reflexivity
        | exact Hconts ].
  - (* Let_cont_recursive *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
    split; [ | exact Hagrees ].
    destruct He as [-> | [-> Hno]].
    + simpl fold_known.
      rewrite step_let_cont_recursive, fold_known_handlers_bind_statics,
        Hbinds.
      eexists. split; [ reflexivity | ].
      apply State_kept; [ left; reflexivity | ].
      apply Forall2_app; [ | exact Hconts ].
      apply rec_closures_kept; [ left; reflexivity | exact Hconts ].
    + rewrite step_let_cont_recursive, Hbinds.
      simpl in Hno. apply andb_true_iff in Hno as [Hno_hs Hno_b].
      eexists. split; [ reflexivity | ].
      apply State_kept; [ right; split; [ reflexivity | exact Hno_b ] | ].
      apply Forall2_app; [ | exact Hconts ].
      apply rec_closures_kept;
        [ right; split; [ reflexivity | exact Hno_hs ] | exact Hconts ].
  - (* Apply: the call as it is has, in the second run, the step that
       answers the first; the call folded then has that step. *)
    assert (He' :
      step (State (Apply ap) vars conts' (fold_statics statics) hp rd)
      <> Undefined ->
      step (State e' vars conts' (fold_statics statics) hp rd)
      = step (State (Apply ap) vars conts' (fold_statics statics) hp rd)).
    { destruct He as [-> | [-> _]]; [ | reflexivity ].
      simpl fold_known.
      exact (fold_known_apply_sound_here T (fold_statics statics) hp rd
               (proj2 Hagrees) vars conts' ap). }
    cut
      (match step (State (Apply ap) vars conts statics hp rd) with
       | Step t =>
         (∃ t' : state,
           step (State (Apply ap) vars conts' (fold_statics statics) hp rd)
           = Step t'
           ∧ state_kept t t')
         ∧ agrees (state_statics t)
       | Exit k vs hp' rd' =>
         step (State (Apply ap) vars conts' (fold_statics statics) hp rd)
         = Exit k vs hp' rd'
       | Undefined => True
       end).
    { destruct (step (State (Apply ap) vars conts statics hp rd))
        as [t | k vs hp' rd' | ]; [ | | exact (fun H => H) ].
      - intros [[t' [Ht' Hkept]] Hstatics]. split; [ | exact Hstatics ].
        exists t'. split; [ | exact Hkept ].
        rewrite He' by (rewrite Ht'; discriminate). exact Ht'.
      - intros Hexit. rewrite He' by (rewrite Hexit; discriminate).
        exact Hexit. }
    rewrite !step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    pose proof
      (call_function_kept callee (ap_call_kind ap) vs (ap_args_arity ap)
         (ap_return_arity ap) (resolve (ap_continuation ap) conts)
         (resolve (ap_continuation ap) conts')
         (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
         (with_extra_args (resolve (ap_exn_continuation ap) conts') extra)
         statics hp rd (resolve_kept _ conts conts' Hconts)
         (with_extra_args_kept _ _ extra
            (resolve_kept _ conts conts' Hconts)))
      as Hcall.
    destruct (call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
                (ap_return_arity ap) (resolve (ap_continuation ap) conts)
                (with_extra_args (resolve (ap_exn_continuation ap) conts)
                   extra)
                statics hp rd)
      as [t | k exit_vs | ] eqn:Hstep;
      [ | exact Hcall | exact I ].
    split; [ exact Hcall | ].
    rewrite (call_function_statics _ _ _ _ _ _ _ _ _ _ t Hstep).
    exact Hagrees.
  - (* Apply_cont *)
    assert (He' : e' = Apply_cont ac)
      by (destruct He as [-> | [-> _]]; reflexivity).
    subst e'. rewrite !step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation.
    pose proof
      (apply_cont_value_kept (resolve (ac_continuation ac) conts)
         (resolve (ac_continuation ac) conts')
         (resolve_kept _ conts conts' Hconts) vs statics hp rd)
      as Happly.
    destruct
      (apply_cont_value (resolve (ac_continuation ac) conts) vs statics hp
         rd)
      as [t | | ] eqn:Hstep;
      [ | exact Happly | exact I ].
    split; [ exact Happly | ].
    rewrite (apply_cont_value_statics _ _ _ _ _ t Hstep). exact Hagrees.
  - (* Switch *)
    assert (He' : e' = Switch scrutinee arms)
      by (destruct He as [-> | [-> _]]; reflexivity).
    subst e'. rewrite !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; [ | exact I ].
    split; [ | exact Hagrees ].
    eexists. split; [ reflexivity | ].
    apply State_kept; [ left; reflexivity | exact Hconts ].
  - (* Invalid *)
    rewrite step_invalid. exact I.
Qed.

Definition kept_related (s s' : state) : Prop :=
  state_kept s s' ∧ bound_as_inv s ∧ agrees (state_statics s).

Lemma kept_related_step (s s' : state) :
  kept_related s s' -> result_simulated kept_related (step s) (step s').
Proof.
  intros (Hkept & Hinv & Hagrees).
  destruct Hkept as [e e' vars conts conts' statics hp rd He Hconts].
  pose proof
    (kept_step_components e e' vars conts conts' statics hp rd He Hconts
       (proj1 Hinv) Hagrees)
    as Hstep.
  pose proof (bound_as_inv_step (State e vars conts statics hp rd)) as
    Hkept_inv.
  destruct (step (State e vars conts statics hp rd)) as [t | k vs | ]; simpl;
    [ | | exact I ].
  - destruct Hstep as [(t' & -> & Ht) Hagrees']. simpl.
    split; [ exact Ht | ].
    split; [ exact (Hkept_inv t Hinv eq_refl) | exact Hagrees' ].
  - rewrite Hstep. simpl. auto.
Qed.

End Unit.

(* A symbol of which something is known is one that the expression
   binds, outside code. *)
Lemma known_of_binding_symbol (binding : static_binding) (s : symbol) :
  In s (map fst (known_of_binding binding)) ->
  In s (symbols_of_binding binding).
Proof.
  destruct binding as [id c | id | s1 const]; simpl; try tauto.
  destruct (closed_static_const const); simpl; [ | tauto ].
  destruct (eval_static_const [] const); simpl; tauto.
Qed.

Lemma set_of_binding_symbol (binding : static_binding) (s : symbol) :
  In s (map fst (set_of_binding binding)) ->
  In s (symbols_of_binding binding).
Proof.
  destruct binding as [id c | id | s1 const]; simpl; try tauto.
  destruct const; simpl; try tauto.
  repeat
    match goal with
    | |- context [match ?x with _ => _ end] => destruct x; simpl
    end;
    tauto.
Qed.

Fixpoint known_bindings_symbols (e : expr) (s : symbol) {struct e} :
  In s (map fst (known_bindings e)) ->
  In s (static_names symbols_of_binding e)
with known_bindings_of_handler_symbols (h : continuation_handler)
    (s : symbol) {struct h} :
  In s (map fst (known_bindings_of_handler h)) ->
  In s (handler_static_names symbols_of_binding h).
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try tauto.
    + apply known_bindings_symbols.
    + rewrite map_app, !in_app_iff.
      intros [H | H];
        [ left | right; exact (known_bindings_symbols body s H) ].
      induction bindings as [ | binding bindings IH]; simpl in H |- *;
        [ exact H | ].
      rewrite map_app, in_app_iff in H. apply in_or_app.
      destruct H as [H | H];
        [ left; exact (known_of_binding_symbol binding s H)
        | right; exact (IH H) ].
    + rewrite map_app, !in_app_iff.
      intros [H | H];
        [ left; exact (known_bindings_of_handler_symbols h s H)
        | right; exact (known_bindings_symbols body s H) ].
    + rewrite map_app, !in_app_iff.
      intros [H | H];
        [ left | right; exact (known_bindings_symbols body s H) ].
      induction handlers as [ | [k1 h1] handlers IH]; simpl in H |- *;
        [ exact H | ].
      rewrite map_app, in_app_iff in H. apply in_or_app.
      destruct H as [H | H];
        [ left; exact (known_bindings_of_handler_symbols h1 s H)
        | right; exact (IH H) ].
  - destruct h as [params metadata handler]. simpl.
    apply known_bindings_symbols.
Qed.

Fixpoint known_sets_symbols (e : expr) (s : symbol) {struct e} :
  In s (map fst (known_sets e)) -> In s (static_names symbols_of_binding e)
with known_sets_of_handler_symbols (h : continuation_handler) (s : symbol)
    {struct h} :
  In s (map fst (known_sets_of_handler h)) ->
  In s (handler_static_names symbols_of_binding h).
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try tauto.
    + apply known_sets_symbols.
    + rewrite map_app, !in_app_iff.
      intros [H | H]; [ left | right; exact (known_sets_symbols body s H) ].
      induction bindings as [ | binding bindings IH]; simpl in H |- *;
        [ exact H | ].
      rewrite map_app, in_app_iff in H. apply in_or_app.
      destruct H as [H | H];
        [ left; exact (set_of_binding_symbol binding s H)
        | right; exact (IH H) ].
    + rewrite map_app, !in_app_iff.
      intros [H | H];
        [ left; exact (known_sets_of_handler_symbols h s H)
        | right; exact (known_sets_symbols body s H) ].
    + rewrite map_app, !in_app_iff.
      intros [H | H]; [ left | right; exact (known_sets_symbols body s H) ].
      induction handlers as [ | [k1 h1] handlers IH]; simpl in H |- *;
        [ exact H | ].
      rewrite map_app, in_app_iff in H. apply in_or_app.
      destruct H as [H | H];
        [ left; exact (known_sets_of_handler_symbols h1 s H)
        | right; exact (IH H) ].
  - destruct h as [params metadata handler]. simpl.
    apply known_sets_symbols.
Qed.

Lemma known_statics_symbol (known : list (symbol * static_const))
    (s : symbol) (allocated : statically_allocated_value) :
  lookup s (static_values (known_statics known)) = Some allocated ->
  In s (map fst known).
Proof.
  intros H. apply lookup_in in H. unfold known_statics in H.
  cbn [static_values] in H.
  apply in_flat_map in H as [[s1 const] [Hin H]].
  unfold known_value in H. cbn [fst snd] in H.
  destruct (eval_static_const [] const); [ | destruct H ].
  destruct H as [[= <- _] | []].
  apply in_map_iff. exists (s1, const). split; [ reflexivity | exact Hin ].
Qed.

(* A static environment that binds no name that the unit binds agrees
   with what is known of the unit: it binds none of its symbols. *)
Lemma agrees_from (e : expr) (statics : static_env) :
  static_names_free_in statics e = true ->
  agrees (known_bindings e) (known_sets e) statics.
Proof.
  intros Hfree. split; intros s allocated allocated' Hknown Hrun; exfalso.
  - pose proof
      (free_in_symbols statics e Hfree s
         (known_bindings_symbols e s
            (known_statics_symbol _ s allocated Hknown)))
      as Hunbound.
    unfold symbol_bound in Hunbound. rewrite Hrun in Hunbound.
    discriminate Hunbound.
  - pose proof
      (free_in_symbols statics e Hfree s
         (known_sets_symbols e s (known_statics_symbol _ s allocated Hknown)))
      as Hunbound.
    unfold symbol_bound in Hunbound. rewrite Hrun in Hunbound.
    discriminate Hunbound.
Qed.

(* A term with no static binding in it binds every known symbol as is
   known: it binds none. *)
Fixpoint no_static_bound_as (known sets : list (symbol * static_const))
    (e : expr) {struct e} :
  binds_no_static e = true -> bound_as known sets e = true.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; reflexivity).
  - apply no_static_bound_as.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros H.
    apply andb_true_iff in H as [Hh Hb].
    rewrite (no_static_bound_as known sets handler Hh),
      (no_static_bound_as known sets body Hb).
    reflexivity.
  - intros H. apply andb_true_iff in H as [Hhs Hb].
    rewrite (no_static_bound_as known sets body Hb), andb_true_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhs. apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (no_static_bound_as known sets handler Hh). exact (IH Hrest).
Qed.

(* What may follow a unit is left as it is, and binds the known symbols
   as is known. *)
Lemma observers_kept (known sets : list (symbol * static_const))
    (slots : list value_slot) (conts : cont_env) :
  observers slots conts ->
  conts_kept known sets conts conts
  ∧ conts_terms_ok (handler_bound_as known sets) conts.
Proof.
  unfold observers, conts_kept, conts_terms_ok.
  induction conts as [ | [k c] conts IH]; simpl;
    [ intros _; split; [ constructor | exact I ] | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  destruct (IH Hrest) as [Hkept Hok].
  assert (Hthis :
    cont_kept known sets c c
    ∧ cont_terms_ok (handler_bound_as known sets) c).
  { revert Hc.
    destruct c
      as [ [params metadata body] vars kept recursive | k0 | c0 extra
         | remaining remaining_arity return_arity return_to exn_to ];
      simpl; try contradiction;
      [ | intros _; split; [ apply Outer_kept | exact I ] ].
    destruct vars; [ | contradiction ].
    destruct kept; [ | contradiction ].
    destruct recursive; [ | contradiction ].
    intros (Hno & _ & _). split.
    - apply Closure_kept;
        [ right; split; [ reflexivity | exact Hno ] | left; reflexivity
        | constructor ].
    - split; [ exact (no_static_bound_as known sets body Hno) | ].
      split; [ reflexivity | exact I ]. }
  destruct Hthis as [Hc_kept Hc_ok].
  split;
    [ constructor; [ split; [ reflexivity | exact Hc_kept ] | exact Hkept ]
    | exact (conj Hc_ok Hok) ].
Qed.

(* What [fold_known_unit] gives refines the unit, the two run from a
   static environment that holds no code and binds no name that the
   unit binds, from any heap, and from a continuation environment of
   what may follow a unit. *)
Lemma fold_known_unit_refines_under (slots : list value_slot) (e e' : expr)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  fold_known_unit e = Some e' ->
  static_code statics = [] ->
  static_names_free_in statics e = true ->
  observers slots conts ->
  refines (State e [] conts statics hp rd) (State e' [] conts statics hp rd).
Proof.
  unfold fold_known_unit. cbv zeta.
  destruct (bound_as (known_bindings e) (known_sets e) e) eqn:Hbound;
    [ | discriminate ].
  intros [= <-] Hno_code Hfree Hobservers.
  destruct (observers_kept (known_bindings e) (known_sets e) slots conts
              Hobservers)
    as [Hkept Hok].
  apply (simulation_refines
           (kept_related (known_bindings e) (known_sets e))
           (kept_related_step (known_bindings e)
              (known_bindings_have_values e) (known_sets e)
              (known_sets_have_values e))).
  split.
  - destruct statics as [code values]. simpl in Hno_code. subst code.
    exact (State_kept (known_bindings e) (known_sets e) e _ [] conts conts
             (Static_env [] values) hp rd_ (or_introl eq_refl) Hkept).
  - split; [ | exact (agrees_from e statics Hfree) ].
    unfold bound_as_inv, terms_inv. simpl_state.
    refine (conj Hbound (conj Hok _)).
    intros id c Hlookup. rewrite Hno_code in Hlookup. discriminate Hlookup.
Qed.

Lemma fold_known_unit_refines_from (e e' : expr) (statics : static_env)
    (hp : heap) (rd : read_record) :
  fold_known_unit e = Some e' ->
  static_code statics = [] ->
  static_names_free_in statics e = true ->
  refines (State e [] [] statics hp rd) (State e' [] [] statics hp rd).
Proof.
  intros Hfold Hno_code Hfree.
  exact (fold_known_unit_refines_under [] e e' [] statics hp rd Hfold Hno_code
           Hfree (Forall_nil _)).
Qed.

Theorem fold_known_unit_program_refines :
  fold_known_unit_program_refines_statement.
Proof.
  intros e e' Hfold. unfold program_refines.
  exact
    (fold_known_unit_refines_from e e' empty_static_env empty_heap no_reads
       Hfold
           eq_refl (free_in_empty e)).
Qed.

Theorem fold_known_unit_refines_imports :
  fold_known_unit_refines_imports_statement.
Proof.
  intros imports e e' Hfold Hfree. unfold unit_refines.
  exact (fold_known_unit_refines_from e e' (imports_statics imports)
           empty_heap no_reads Hfold eq_refl Hfree).
Qed.

Theorem fold_known_unit_refines_observed :
  fold_known_unit_refines_observed_statement.
Proof.
  intros imports conts e e' Hfold Hfree Hobservers.
  exact (fold_known_unit_refines_under [] e e' conts
           (imports_statics imports) empty_heap no_reads Hfold eq_refl Hfree
           Hobservers).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The law of the pass is proved from the definitions by [equivariance]
   (FlambdaEquivarianceTactic.v), and so are the laws of the tests and of
   the three folds that read a symbol. Three lemmas stay by hand: a
   simple or a defining expression that is known mentions no variable,
   so renaming leaves it as it is, and what it folds to with it. That is
   a fact about what passed a test, not a law of renaming; the third is
   given to the tactic to rewrite with, as are the laws of the folds. *)

(* The database is applied by its one pass, the faster step: every
   proof below passes with it. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

Lemma rename_known_simple (r : renaming) (statics : static_env) (s : simple) :
  simple_known statics s = true -> rename_simple r s = s.
Proof. destruct s; simpl; [ discriminate | reflexivity | reflexivity ]. Qed.

Lemma simple_known_rename (r : renaming) (statics : static_env) (s : simple) :
  simple_known statics (rename_simple r s) = simple_known statics s.
Proof. equivariance. Qed.

Lemma named_known_rename (r : renaming) (statics : static_env) (d : named) :
  named_known statics (rename_named r d) = named_known statics d.
Proof. equivariance. Qed.

Lemma rename_known_named (r : renaming) (statics : static_env) (d : named) :
  named_known statics d = true -> rename_named r d = d.
Proof.
  destruct d; simpl; try discriminate.
  - intros H. apply andb_true_iff in H as [_ Hargs]. f_equal.
    induction args as [ | s args IH]; simpl; [ reflexivity | ].
    simpl in Hargs. apply andb_true_iff in Hargs as [Hs Hargs].
    rewrite (rename_known_simple r statics s Hs), (IH Hargs). reflexivity.
  - intros H. rewrite (rename_known_simple r statics _ H). reflexivity.
  - intros H. rewrite (rename_known_simple r statics _ H). reflexivity.
  - intros H. rewrite (rename_known_simple r statics _ H). reflexivity.
Qed.

Lemma fold_known_named_rename (r : renaming) (statics : static_env)
    (d : named) :
  rename_named r (fold_known_named statics d)
  = fold_known_named statics (rename_named r d).
Proof.
  unfold fold_known_named. rewrite named_known_rename.
  destruct (named_known statics d) eqn:Hknown; [ | reflexivity ].
  rewrite (rename_known_named r statics d Hknown).
  destruct (eval_named statics [] d) as [[] | ]; try reflexivity;
    exact (rename_known_named r statics d Hknown).
Qed.
#[local] Hint Rewrite fold_known_named_rename : equivariance.

(* The projection of a function slot that is folded is from a symbol. *)
Lemma fold_function_slot_rename (r : renaming) (sets : static_env)
    (d : named) :
  rename_named r (fold_function_slot sets d)
  = fold_function_slot sets (rename_named r d).
Proof. equivariance. Qed.
#[local] Hint Rewrite fold_function_slot_rename : equivariance.

(* The load of a field that is folded is from a symbol. *)
Lemma fold_block_load_rename (r : renaming) (sets : static_env) (d : named) :
  rename_named r (fold_block_load sets d)
  = fold_block_load sets (rename_named r d).
Proof. equivariance. Qed.
#[local] Hint Rewrite fold_block_load_rename : equivariance.

Lemma fold_known_apply_rename (r : renaming) (sets : static_env)
    (ap : apply_expr) :
  rename_apply r (fold_known_apply sets ap)
  = fold_known_apply sets (rename_apply r ap).
Proof. equivariance. Qed.
#[local] Hint Rewrite fold_known_apply_rename : equivariance.

Theorem fold_known_equivariant : fold_known_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Theorem fold_known_apply_defines : fold_known_apply_defines_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ exists 20; vm_compute; reflexivity | ].
  exists 20; eexists; vm_compute; reflexivity.
Qed.
