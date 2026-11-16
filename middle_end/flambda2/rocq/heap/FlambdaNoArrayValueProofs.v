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

(* Proofs of the statements of FlambdaNoArrayValueSpec.v. Nothing here
   needs review.

   The invariance of the machine follows FlambdaRangeProofs.v, which
   keeps another property of whole states along a run. The primitives
   are not looked into: what one that makes no array does to values
   that hold none is
   FlambdaPrimitiveProofs.eval_prim_preserves_making_no_array. The
   machine with a heap takes the step of the machine where no primitive
   is run (FlambdaHeapMachineProofs.heap_step_agrees). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaAlphaSpec
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaHeapSpec
  FlambdaHeapProofs FlambdaHeapMachineSpec FlambdaHeapMachineProofs
  FlambdaHeapEquivarianceSpec FlambdaHeapEquivarianceProofs FlambdaLiftSpec
  FlambdaLiftProofs FlambdaWellFormedSpec FlambdaMyClosureSpec
  FlambdaEquivarianceTactic FlambdaNoArrayValueSpec.
Import ListNotations.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

Definition slots_no_array_value (slots : list (value_slot * value)) : bool :=
  forallb (fun slot => no_array_value (snd slot)) slots.

Lemma forallb_of_map {A B : Type} (f : B -> bool) (g : A -> B)
    (l : list A) :
  forallb f (map g l) = forallb (fun a => f (g a)) l.
Proof.
  induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma lookup_slot_no_array_value (slot : value_slot)
    (value_slots : list (value_slot * value)) (v : value) :
  slots_no_array_value value_slots = true ->
  lookup slot value_slots = Some v ->
  no_array_value v = true.
Proof.
  intros Hslots Hlookup.
  exact
    (proj1 (forallb_forall _ _) Hslots (slot, v) (lookup_in _ _ _ Hlookup)).
Qed.

(* No array value in a partial application: none in the value it calls
   nor in its arguments. *)
Lemma no_array_value_partial (callee : value) (args : list value)
    (args_arity : arity) :
  no_array_value (Value_partial_application callee args args_arity) = true
  <-> no_array_value callee = true
      ∧ Forall (fun v => no_array_value v = true) args.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

(* ======================================================================== *)
(* The observation                                                          *)
(* ======================================================================== *)

Theorem array_of_no_array_value : array_of_no_array_value_statement.
Proof.
  intros statics v elements Hv. split.
  - intros Harray.
    destruct (array_of_cases statics v elements Harray)
      as [-> | Hsymbol]; [ discriminate Hv | exact Hsymbol ].
  - intros (s & -> & Hlookup). simpl. rewrite Hlookup. reflexivity.
Qed.

(* ======================================================================== *)
(* Evaluation                                                               *)
(* ======================================================================== *)

Section Evaluation.

Variable vars : env.
Hypothesis Hvars : env_no_array_value vars.

Lemma eval_simple_no_array_value (s : simple) (v : value) :
  eval_simple vars s = Some v -> no_array_value v = true.
Proof.
  destruct s as [x | sym | c]; simpl.
  - intros Hlookup. exact (Hvars x v (lookup_in _ _ _ Hlookup)).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

Lemma eval_simples_no_array_value (args : list simple) :
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    forallb no_array_value vs = true.
Proof.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_no_array_value s v Heval), (IH vs' eq_refl).
    reflexivity.
Qed.

Lemma eval_value_slots_no_array_value
    (value_slots : list (value_slot * simple)) :
  ∀ evaluated : list (value_slot * value),
    eval_value_slots vars value_slots = Some evaluated ->
    slots_no_array_value evaluated = true.
Proof.
  unfold slots_no_array_value.
  induction value_slots as [ | [slot s] value_slots IH]; intros evaluated;
    simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_no_array_value s v Heval), (IH rest eq_refl).
    reflexivity.
Qed.

End Evaluation.

(* ======================================================================== *)
(* What is read through the static environment                              *)
(* ======================================================================== *)

Section Statics.

Variable statics : static_env.
Hypothesis Hstatics : statics_no_array_value statics.

Lemma lookup_symbol_no_array_value (s : symbol)
    (allocated : statically_allocated_value) :
  lookup s (static_values statics) = Some allocated ->
  static_no_array_value allocated = true.
Proof.
  intros Hlookup. exact (Hstatics s allocated (lookup_in _ _ _ Hlookup)).
Qed.

(* The value slots of a closure that holds no array value hold none,
   whether the closure was made by a [Let] or is what a symbol is bound
   to. *)
Lemma closure_in_set_no_array_value (v : value) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  no_array_value v = true ->
  closure_in_set statics v = Some (slot, function_decls, value_slots) ->
  slots_no_array_value value_slots = true.
Proof.
  intros Hv. destruct v; simpl; try discriminate.
  - intros [= _ _ <-]. exact Hv.
  - match goal with
    | |- context [lookup ?at_slot ?decls] => destruct (lookup at_slot decls)
    end;
      [ | discriminate ].
    simpl. intros [= _ _ <-]. exact Hv.
  - match goal with
    | |- context [lookup ?s ?values] =>
      destruct (lookup s values) as [allocated | ] eqn:Hlookup;
        [ | discriminate ];
        pose proof (lookup_symbol_no_array_value s allocated Hlookup)
          as Hallocated
    end.
    unfold static_no_array_value in Hallocated.
    destruct allocated; try discriminate; simpl in Hallocated;
      rewrite forallb_of_map in Hallocated.
    + intros [= _ _ <-]. exact Hallocated.
    + match goal with
      | |- context [lookup ?at_slot ?decls] =>
        destruct (lookup at_slot decls)
      end;
        [ | discriminate ].
      simpl. intros [= _ _ <-]. exact Hallocated.
Qed.

(* The closure of the same set at another slot has the same value slots,
   and a symbol holds no array value. *)
Lemma project_function_slot_no_array_value
    (move_from move_to : function_slot) (v w : value) :
  no_array_value v = true ->
  project_function_slot statics move_from move_to v = Some w ->
  no_array_value w = true.
Proof.
  intros Hv. unfold project_function_slot.
  destruct v; try discriminate.
  - destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. exact Hv.
  - match goal with
    | |- context [Pos.eqb ?at_slot move_from] =>
      destruct (Pos.eqb at_slot move_from); [ | discriminate ]
    end.
    repeat
      match goal with
      | |- context [lookup ?at_slot ?decls] =>
        destruct (lookup at_slot decls); [ | discriminate ]
      end.
    simpl. intros [= <-]. exact Hv.
  - match goal with
    | |- context [lookup ?s ?values] =>
      destruct (lookup s values) as [allocated | ]; [ | discriminate ]
    end.
    destruct allocated; try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros [= <-]. reflexivity.
    + match goal with
      | |- context [Pos.eqb ?at_slot move_from] =>
        destruct (Pos.eqb at_slot move_from); [ | discriminate ]
      end.
      repeat
        match goal with
        | |- context [lookup ?key ?bindings] =>
          destruct (lookup key bindings); [ | discriminate ]
        end.
      simpl. intros [= <-]. reflexivity.
Qed.

(* The fields of a block that holds no array value hold none. *)
Lemma block_of_no_array_value (v : value) (tag : Z) (fields : list value) :
  no_array_value v = true ->
  block_of statics v = Some (tag, fields) ->
  forallb no_array_value fields = true.
Proof.
  intros Hv Hblock.
  destruct (block_of_cases _ _ _ _ Hblock) as [-> | (s & -> & Hs)].
  - exact Hv.
  - exact (lookup_symbol_no_array_value s _ Hs).
Qed.

(* The elements of an array that is no array value hold none. *)
Lemma array_of_elements_no_array_value (v : value)
    (elements : list value) :
  no_array_value v = true ->
  array_of statics v = Some elements ->
  forallb no_array_value elements = true.
Proof.
  intros Hv Harray.
  destruct (array_of_cases statics v elements Harray)
    as [-> | (s & -> & Hs)].
  - discriminate Hv.
  - exact (lookup_symbol_no_array_value s _ Hs).
Qed.

Theorem eval_prim_no_array_value_here (p : prim) (vs : list value)
    (v : value) :
  prim_makes_array p = false ->
  (∀ arg : value, arg ∈ vs -> no_array_value arg = true) ->
  eval_prim statics p vs = Some v ->
  no_array_value v = true.
Proof.
  intros Hp Hvs.
  apply
    (eval_prim_preserves_making_no_array (fun w => no_array_value w = true)
       statics p vs v).
  - intros kind ->. discriminate Hp.
  - intros c. split; reflexivity.
  - intros tag fields. simpl. apply forallb_forall.
  - intros elements Harray. discriminate Harray.
  - intros s tag fields _ Hlookup. apply forallb_forall.
    exact (lookup_symbol_no_array_value s _ Hlookup).
  - intros s elements _ Hlookup. apply forallb_forall.
    exact (lookup_symbol_no_array_value s _ Hlookup).
  - exact Hvs.
Qed.

Variable vars : env.
Hypothesis Hvars : env_no_array_value vars.

Lemma eval_named_no_array_value (d : named) (v : value) :
  named_makes_array d = false ->
  eval_named statics vars d = Some v ->
  no_array_value v = true.
Proof.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure];
    simpl; intros Hd.
  - exact (eval_simple_no_array_value vars Hvars s v).
  - destruct (eval_simples vars args) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl.
    pose proof (eval_simples_no_array_value vars Hvars args vs Heval) as Hvs.
    apply (eval_prim_no_array_value_here p vs v Hd).
    apply forallb_forall. exact Hvs.
  - destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_value_slots_no_array_value vars Hvars slots evaluated Heval).
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_value_slots_no_array_value vars Hvars slots evaluated Heval).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    simpl.
    destruct (closure_of statics w) as [[called value_slots] | ] eqn:Hof;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics w called value_slots) Hof)
      as (at_slot & function_decls & Hin_set & _).
    exact
      (lookup_slot_no_array_value slot value_slots v
         (closure_in_set_no_array_value w at_slot function_decls value_slots
            (eval_simple_no_array_value vars Hvars closure w Hclosure)
            Hin_set)
         Hlookup).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hproject.
    destruct (proj1 (project_value_slot_defined statics pfrom slot w v)
                Hproject)
      as (function_decls & value_slots & Hin_set & Hlookup).
    exact
      (lookup_slot_no_array_value slot value_slots v
         (closure_in_set_no_array_value w pfrom function_decls value_slots
            (eval_simple_no_array_value vars Hvars closure w Hclosure)
            Hin_set)
         Hlookup).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    simpl.
    exact
      (project_function_slot_no_array_value move_from move_to w v
         (eval_simple_no_array_value vars Hvars closure w Hclosure)).
Qed.

End Statics.

Theorem eval_prim_no_array_value : eval_prim_no_array_value_statement.
Proof.
  intros statics p vs v Hp Hstatics Hvs.
  exact (eval_prim_no_array_value_here statics Hstatics p vs v Hp Hvs).
Qed.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

Lemma eval_static_const_no_array_value (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  env_no_array_value vars ->
  eval_static_const vars const = Some allocated ->
  static_no_array_value allocated = true.
Proof.
  intros Hvars. unfold static_no_array_value.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite forallb_of_map.
    exact (eval_value_slots_no_array_value vars Hvars slots evaluated Heval).
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite forallb_of_map.
    exact (eval_value_slots_no_array_value vars Hvars slots evaluated Heval).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_no_array_value vars Hvars fields vs Heval).
  - destruct (eval_simple vars contents) as [v | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_no_array_value vars Hvars contents v Heval).
    reflexivity.
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_no_array_value vars Hvars elements vs Heval).
Qed.

Lemma eval_static_bindings_free_of_arrays (vars : env)
    (bindings : list static_binding) :
  env_no_array_value vars ->
  ∀ statics statics' : static_env,
    existsb binding_makes_array bindings = false ->
    code_makes_no_array statics ->
    statics_no_array_value statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    code_makes_no_array statics' ∧ statics_no_array_value statics'.
Proof.
  intros Hvars.
  induction bindings as [ | binding bindings IH]; intros statics statics';
    simpl.
  - intros _ Hcode Hvalues [= <-]. split; assumption.
  - intros Hbindings. apply orb_false_iff in Hbindings as [Hbinding Hrest].
    intros Hcode Hvalues.
    destruct binding as [id c | id | s const]; simpl in Hbinding.
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      apply (IH _ _ Hrest); [ | exact Hvalues ].
      intros id' c' [[= <- <-] | Hin]; [ exact Hbinding | ].
      exact (Hcode id' c' Hin).
    + apply (IH _ _ Hrest); assumption.
    + destruct (lookup s (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Heval;
        [ | discriminate ].
      simpl. apply (IH _ _ Hrest); [ exact Hcode | ].
      intros s' allocated' [[= <- <-] | Hin].
      * exact (eval_static_const_no_array_value vars const allocated Hvars
                 Heval).
      * exact (Hvalues s' allocated' Hin).
Qed.

(* ======================================================================== *)
(* Continuations and calls                                                  *)
(* ======================================================================== *)

Lemma resolve_free_of_arrays (k : continuation) (conts : cont_env) :
  conts_free_of_arrays conts -> cont_free_of_arrays (resolve k conts).
Proof.
  intros Hconts. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hlookup.
  - exact (Hconts k c (lookup_in _ _ _ Hlookup)).
  - constructor.
Qed.

Lemma handlers_make_no_array
    (handlers : list (continuation * continuation_handler)) :
  existsb (fun '(_, h) => handler_makes_array h) handlers = false ->
  ∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers -> handler_makes_array h = false.
Proof.
  intros Hhandlers k h Hin.
  destruct (handler_makes_array h) eqn:Hh; [ | reflexivity ].
  assert (Hsome :
    existsb (fun '(_, h) => handler_makes_array h) handlers = true).
  { apply existsb_exists. exists (k, h). split; [ exact Hin | exact Hh ]. }
  congruence.
Qed.

Lemma rec_closures_free_of_arrays
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  (∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers -> handler_makes_array h = false) ->
  env_no_array_value vars ->
  conts_free_of_arrays conts ->
  conts_free_of_arrays (rec_closures handlers vars conts).
Proof.
  intros Hhandlers Hvars Hconts k c Hin. unfold rec_closures in Hin.
  apply in_map_iff in Hin as ([k' h] & [= <- <-] & Hin).
  constructor; [ exact (Hhandlers k' h Hin) | assumption .. ].
Qed.

Lemma conts_free_of_arrays_app (conts conts' : cont_env) :
  conts_free_of_arrays conts ->
  conts_free_of_arrays conts' ->
  conts_free_of_arrays (conts ++ conts').
Proof.
  intros Hconts Hconts' k c Hin.
  apply in_app_or in Hin as [Hin | Hin]; eauto.
Qed.

Lemma bind_params_no_array_value (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    forallb no_array_value vs = true ->
    env_no_array_value vars ->
    bind_params params vs vars = Some vars' ->
    env_no_array_value vars'.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars vars'; simpl;
    try discriminate.
  - intros _ Hvars [= <-]. exact Hvars.
  - intros Hvs. apply andb_true_iff in Hvs as [Hv Hvs]. intros Hvars.
    apply (IH vs _ _ Hvs).
    intros y w [[= <- <-] | Hin]; [ exact Hv | exact (Hvars y w Hin) ].
Qed.

(* What a step comes to: a state free of arrays, or a way out with
   values and a heap that hold no array value. *)
Definition result_free_of_arrays (result : step_result) : Prop :=
  match result with
  | Step s => state_free_of_arrays s
  | Exit _ vs hp rd =>
    forallb no_array_value vs = true ∧ heap_no_array_value hp
  | Undefined => True
  end.

(* The fields of a tuple that holds no array value hold none. *)
Lemma tuple_fields_no_array_value (statics : static_env) :
  statics_no_array_value statics ->
  P_fields (fun v => no_array_value v = true) statics.
Proof.
  intros Hstatics size tuple fields Htuple Hfields.
  apply (tuple_fields_of_blocks _ statics size tuple fields);
    [ | exact Hfields ].
  intros tag block Hblock. apply forallb_Forall.
  exact (block_of_no_array_value statics Hstatics tuple tag block Htuple
           Hblock).
Qed.

Section Calls.

Variable statics : static_env.
Variable hp : heap.
Variable rd : read_record.
Hypothesis Hcode : code_makes_no_array statics.
Hypothesis Hstatics : statics_no_array_value statics.
Hypothesis Hheap : heap_no_array_value hp.

Lemma call_code_free_of_arrays (closure : value) (called : code_id)
    (c : code) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) :
  no_array_value closure = true ->
  forallb no_array_value vs = true ->
  cont_free_of_arrays return_to ->
  cont_free_of_arrays exn_to ->
  lookup called (static_code statics) = Some c ->
  result_free_of_arrays
    (call_code closure c vs args_arity return_arity return_to exn_to statics
       hp rd).
Proof.
  intros Hclosure Hvs Hreturn Hexn Hlookup. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ] eqn:Hbind;
    [ | exact I ].
  pose proof (Hcode called c (lookup_in _ _ _ Hlookup)) as Hc.
  repeat split; simpl.
  - destruct c. exact Hc.
  - apply (bind_params_no_array_value _ _ _ _ Hvs) in Hbind;
      [ exact Hbind | ].
    intros y w [[= <- <-] | []]. exact Hclosure.
  - intros k c' [[= <- <-] | [[= <- <-] | []]]; assumption.
  - exact Hcode.
  - exact Hstatics.
  - exact Hheap.
Qed.

Lemma run_call_free_of_arrays (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) :
  shape_calls statics kind shape ->
  shape_values_all (fun v => no_array_value v = true) shape ->
  cont_free_of_arrays return_to ->
  cont_free_of_arrays exn_to ->
  (∀ partial : value,
    no_array_value partial = true ->
    result_free_of_arrays (return_partial partial)) ->
  result_free_of_arrays
    (run_call return_partial shape return_arity return_to exn_to statics hp rd).
Proof.
  intros Hcalls Hvalues Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls, Hvalues |- *.
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    destruct Hvalues as [Hclosure Hvs]. apply forallb_Forall in Hvs.
    exact (call_code_free_of_arrays closure called c vs _ _ _ _ Hclosure
             Hvs Hreturn Hexn Hlookup).
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    destruct Hvalues as (Hclosure & Hvs & Hremaining).
    apply forallb_Forall in Hvs, Hremaining.
    apply (call_code_free_of_arrays closure called c vs _ _ _ _ Hclosure
             Hvs);
      [ constructor; assumption | exact Hexn | exact Hlookup ].
  - exact (Hpartial partial Hvalues).
  - exact I.
Qed.

Lemma apply_cont_value_free_of_arrays (c : cont_value) (vs : list value) :
  cont_free_of_arrays c ->
  forallb no_array_value vs = true ->
  result_free_of_arrays (apply_cont_value c vs statics hp rd).
Proof.
  intros Hc. revert vs.
  induction Hc
    as [k | c extra Hc IH Hextra
       | [params metadata handler] vars conts recursive Hh Hvars Hconts _
           Hrecursive
       | remaining remaining_arity return_arity return_to exn_to Hremaining
           Hreturn IHreturn Hexn _];
    intros vs Hvs.
  - split; [ exact Hvs | exact Hheap ].
  - simpl. apply IH. rewrite forallb_app, Hvs, Hextra. reflexivity.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [handler_vars | ] eqn:Hbind;
      [ | exact I ].
    repeat split; simpl.
    + exact Hh.
    + exact
        (bind_params_no_array_value params vs vars handler_vars Hvs Hvars
           Hbind).
    + apply conts_free_of_arrays_app; [ | exact Hconts ].
      exact
        (rec_closures_free_of_arrays recursive vars conts Hrecursive Hvars
           Hconts).
    + exact Hcode.
    + exact Hstatics.
    + exact Hheap.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    simpl in Hvs. rewrite andb_true_r in Hvs. unfold call_function.
    apply (run_call_free_of_arrays _ Indirect); try assumption.
    + apply shape_of_call_calls.
    + apply (shape_of_call_values_all _ no_array_value_partial);
        [ exact (tuple_fields_no_array_value statics Hstatics) | exact Hvs
        | apply forallb_Forall; exact Hremaining ].
    + intros partial Hpartial. apply IHreturn. simpl.
      rewrite Hpartial. reflexivity.
Qed.

Lemma with_extra_args_free_of_arrays (c : cont_value) (extra : list value) :
  cont_free_of_arrays c ->
  forallb no_array_value extra = true ->
  cont_free_of_arrays (with_extra_args c extra).
Proof.
  intros Hc Hextra. destruct extra as [ | v extra]; simpl;
    [ exact Hc | constructor; assumption ].
Qed.

Lemma call_function_free_of_arrays (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) :
  no_array_value callee = true ->
  forallb no_array_value vs = true ->
  cont_free_of_arrays return_to ->
  cont_free_of_arrays exn_to ->
  result_free_of_arrays
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd).
Proof.
  intros Hcallee Hvs Hreturn Hexn. unfold call_function.
  apply (run_call_free_of_arrays _ kind); try assumption.
  - apply shape_of_call_calls.
  - apply (shape_of_call_values_all _ no_array_value_partial);
      [ exact (tuple_fields_no_array_value statics Hstatics) | exact Hcallee
      | apply forallb_Forall; exact Hvs ].
  - intros partial Hpartial.
    apply apply_cont_value_free_of_arrays; [ exact Hreturn | ].
    simpl. rewrite Hpartial. reflexivity.
Qed.

End Calls.

(* ======================================================================== *)
(* The primitives on a heap                                                 *)
(* ======================================================================== *)

Lemma naked_bool_no_array_value (b : bool) :
  no_array_value (naked_bool b) = true.
Proof. reflexivity. Qed.

Lemma heap_cons_no_array_value (h : heap) (l : location) (c : cell) :
  heap_no_array_value h ->
  cell_no_array_value c = true ->
  heap_no_array_value ((l, c) :: h).
Proof.
  intros Hh Hc l' c' [[= <- <-] | Hin]; [ exact Hc | exact (Hh l' c' Hin) ].
Qed.

(* By the cases of FlambdaHeapProofs.exec_prim_closed. *)
Theorem exec_prim_no_array_value : exec_prim_no_array_value_statement.
Proof.
  intros statics h h1 p vs result Hp Hstatics Hheap Hvs Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec]. revert Hexec.
  assert (Hpure :
            (let* v := eval_prim statics p vs in Some (v, h))
            = Some (result, h1) ->
            no_array_value result = true ∧ heap_no_array_value h1).
  { destruct (eval_prim statics p vs) as [v | ] eqn:Heval; simpl;
      [ | discriminate ].
    intros [= <- <-].
    split;
      [ exact (eval_prim_no_array_value statics p vs v Hp Hstatics Hvs Heval)
      | exact Hheap ]. }
  destruct p as [p | p | p | p | rp]; try exact Hpure.
  - destruct vs as [ | v [ | v2 vs]]; try exact Hpure.
    2: { destruct v; simpl; discriminate. }
    rewrite exec_bound_prim_unary.
    destruct (location_of v) as [l | ] eqn:Hlocation.
    2: { destruct v; try discriminate; exact Hpure. }
    destruct v; try discriminate. injection Hlocation as ->.
    destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
    destruct (exec_unary_on_cell p l c) as [res | ] eqn:Hres; simpl;
      [ | discriminate ].
    intros [= <- <-]. split; [ | exact Hheap ].
    pose proof (Hheap l c (lookup_in _ _ _ Hc)) as Hcell.
    unfold cell_no_array_value in Hcell.
    destruct c as [cell_mut tag fields | akind elements].
    + destruct p; simpl in Hres; try discriminate.
      * destruct (load_allowed _ cell_mut); [ | discriminate ].
        apply load_field_in in Hres.
        exact (proj1 (forallb_forall _ _) Hcell res Hres).
      * injection Hres as <-. reflexivity.
      * injection Hres as <-. reflexivity.
      * injection Hres as <-. reflexivity.
      * destruct (kind_eqb _ Value); [ | discriminate ].
        injection Hres as <-. reflexivity.
    + destruct p as [ | [ | ] | | | length_kind | | okind | | | | | | | | ];
        simpl in Hres; try discriminate.
      * injection Hres as <-. reflexivity.
      * injection Hres as <-. reflexivity.
      * unfold array_length_of in Hres.
        destruct (in_range _ _); [ | discriminate ].
        injection Hres as <-. reflexivity.
      * destruct (kind_eqb _ Value); [ | discriminate ].
        injection Hres as <-. reflexivity.
  - rewrite exec_bound_prim_binary.
    destruct (phys_equal_on_location p vs).
    + destruct p; try discriminate.
      destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
      destruct (eval_phys_equal_location statics v1 v2); simpl;
        [ | discriminate ].
      intros [= <- <-].
      split; [ apply naked_bool_no_array_value | exact Hheap ].
    + destruct (binary_on_cell p vs) as [[l v2] | ]; [ | exact Hpure ].
      destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
      destruct (exec_binary_on_cell p l c v2) as [res | ] eqn:Hres; simpl;
        [ | discriminate ].
      intros [= <- <-]. split; [ | exact Hheap ].
      pose proof (Hheap l c (lookup_in _ _ _ Hc)) as Hcell.
      unfold cell_no_array_value in Hcell.
      destruct c as [cell_mut tag fields | akind elements];
        [ destruct p; simpl in Hres; discriminate | ].
      destruct p as [array_kind load_kind mut | | | | | ]; simpl in Hres;
        try discriminate.
      destruct (load_allowed mut Cell_mutable); [ | discriminate ].
      apply FlambdaPrimitiveProofs.array_loaded_element in Hres as [Hin _].
      exact (proj1 (forallb_forall _ _) Hcell res Hin).
  - change (exec_bound_prim statics h (Heap p) vs)
      with (exec_heap_primitive statics h p vs).
    destruct p as [mut tag | kind init field | kind | akind set_kind].
    + simpl. destruct (forallb _ vs) eqn:Hkinds; [ | discriminate ].
      intros [= <- <-]. split; [ reflexivity | ].
      apply heap_cons_no_array_value; [ exact Hheap | ].
      apply forallb_forall. exact Hvs.
    + destruct vs as [ | v [ | new [ | v3 vs]]]; try (simpl; discriminate).
      rewrite exec_block_set.
      destruct (location_of v) as [l | ]; [ | discriminate ].
      destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ]
        eqn:Hc;
        try discriminate.
      destruct (store_field kind field new tag fields) as [fields1 | ]
        eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct (store_field_in _ _ _ _ _ _ Hstore) as [_ Hin].
      split; [ reflexivity | ].
      apply heap_cons_no_array_value; [ exact Hheap | ].
      pose proof (Hheap l _ (lookup_in _ _ _ Hc)) as Hcell.
      apply forallb_forall. intros y Hy. destruct (Hin y Hy) as [-> | Hold].
      * apply Hvs. right. left. reflexivity.
      * exact (proj1 (forallb_forall _ _) Hcell y Hold).
    + simpl. destruct vs as [ | v0 vs0]; [ discriminate | ].
      destruct (array_made kind (v0 :: vs0)) as [elements | ] eqn:Hmade;
        simpl; [ | discriminate ].
      intros [= <- <-].
      apply FlambdaPrimitiveProofs.array_made_of_arguments in Hmade
        as [-> _].
      split; [ reflexivity | ].
      apply heap_cons_no_array_value; [ exact Hheap | ].
      apply forallb_forall. exact Hvs.
    + destruct vs as [ | v [ | index [ | new [ | v4 vs]]]];
        try (simpl; discriminate).
      rewrite exec_array_set.
      destruct (location_of v) as [l | ]; [ | discriminate ].
      destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:Hc;
        try discriminate.
      destruct (array_stored set_kind kind elements index new)
        as [stored | ] eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct
        (FlambdaPrimitiveProofs.array_stored_elements _ _ _ _ _ _ Hstore)
        as (_ & _ & Hin).
      split; [ reflexivity | ].
      apply heap_cons_no_array_value; [ exact Hheap | ].
      pose proof (Hheap l _ (lookup_in _ _ _ Hc)) as Hcell.
      apply forallb_forall. intros y Hy. destruct (Hin y Hy) as [-> | Hold].
      * apply Hvs. right. right. left. reflexivity.
      * exact (proj1 (forallb_forall _ _) Hcell y Hold).
Qed.

(* The run of a named expression that makes no array: its value holds
   no array value, and nor does the heap that it leaves. A primitive is
   run by [exec_prim]; anything else has the value that [eval_named]
   gives it and leaves the heap. *)
Lemma exec_named_no_array_value (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  statics_no_array_value statics ->
  env_no_array_value vars ->
  heap_no_array_value hp ->
  named_makes_array d = false ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  no_array_value v = true ∧ heap_no_array_value hp'.
Proof.
  intros Hstatics Hvars Hheap Hd Hexec.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure].
  2: { apply exec_named_prim_some in Hexec as (vs & Hvs & Hprim & _).
       exact
         (exec_prim_no_array_value statics hp hp' p vs v Hd Hstatics Hheap
            (proj1 (forallb_forall _ _)
               (eval_simples_no_array_value vars Hvars args vs Hvs))
            Hprim). }
  all: rewrite exec_named_not_prim in Hexec by (intros ? ?; discriminate);
    destruct (eval_named statics vars _) as [w | ] eqn:Heval;
    [ | discriminate Hexec ];
    injection Hexec as <- <- <-;
    split;
    [ exact
        (eval_named_no_array_value statics Hstatics vars Hvars _ w Hd Heval)
    | exact Hheap ].
Qed.

(* ======================================================================== *)
(* Steps and runs                                                           *)
(* ======================================================================== *)

Lemma step_result_free_of_arrays (s : state) :
  state_free_of_arrays s -> result_free_of_arrays (step s).
Proof.
  intros (Hexpr & Hvars & Hconts & Hcode & Hstatics & Hheap).
  rewrite (state_eta s).
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *. set (hp := state_heap s) in *.
  set (rd := state_reads s) in *.
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step; simpl in Hexpr.
  - apply orb_false_iff in Hexpr as [Hd Hbody].
    destruct (exec_named _ _ _ _ d) as [[[v hp'] rd'] | ] eqn:Heval;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    destruct
      (exec_named_no_array_value statics hp hp' rd rd' vars d v Hstatics
         Hvars Hheap Hd Heval)
      as [Hv Hheap'].
    repeat split; simpl; try assumption.
    intros y w [[= <- <-] | Hin]; [ exact Hv | exact (Hvars y w Hin) ].
  - apply orb_false_iff in Hexpr as [Hbindings Hbody].
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Heval;
      [ | exact I ].
    destruct
      (eval_static_bindings_free_of_arrays vars bindings Hvars statics
         extended Hbindings Hcode Hstatics Heval)
      as [Hcode' Hstatics'].
    repeat split; simpl; assumption.
  - apply orb_false_iff in Hexpr as [Hh Hbody].
    repeat split; simpl; try assumption.
    intros k' c [[= <- <-] | Hin]; [ | exact (Hconts k' c Hin) ].
    constructor; try assumption. intros k' h' [].
  - apply orb_false_iff in Hexpr as [Hhandlers Hbody].
    destruct (handlers_bind_statics handlers); [ exact I | ].
    repeat split; simpl; try assumption.
    apply conts_free_of_arrays_app; [ | exact Hconts ].
    exact
      (rec_closures_free_of_arrays handlers vars conts
         (handlers_make_no_array handlers Hhandlers) Hvars Hconts).
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hc;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextras; [ | exact I ].
    apply call_function_free_of_arrays; try assumption.
    + exact (eval_simple_no_array_value vars Hvars _ callee Hc).
    + exact (eval_simples_no_array_value vars Hvars _ vs Hvs).
    + apply resolve_free_of_arrays, Hconts.
    + apply with_extra_args_free_of_arrays;
        [ apply resolve_free_of_arrays, Hconts | ].
      exact (eval_simples_no_array_value vars Hvars _ extra Hextras).
  - destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    apply apply_cont_value_free_of_arrays; try assumption.
    + apply resolve_free_of_arrays, Hconts.
    + exact (eval_simples_no_array_value vars Hvars _ vs Hvs).
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ | exact I ].
    repeat split; simpl; assumption.
  - exact I.
Qed.

Lemma result_free_of_arrays_statement_form (result : step_result) :
  result_free_of_arrays result ->
  match result with
  | Step s' => state_free_of_arrays s'
  | Exit _ vs hp rd =>
    (∀ v : value, v ∈ vs -> no_array_value v = true)
    ∧ heap_no_array_value hp
  | Undefined => True
  end.
Proof.
  destruct result as [s' | k vs hp rd | ]; simpl; try exact (fun H => H).
  intros [Hvs Hhp]. split; [ apply forallb_forall; exact Hvs | exact Hhp ].
Qed.

Theorem step_free_of_arrays : step_free_of_arrays_statement.
Proof.
  intros s Hs.
  exact
    (result_free_of_arrays_statement_form _ (step_result_free_of_arrays s Hs)).
Qed.

Lemma run_result_free_of_arrays (n : nat) :
  ∀ s : state, state_free_of_arrays s -> result_free_of_arrays (run n s).
Proof.
  induction n as [ | n IH]; intros s Hs; simpl; [ exact Hs | ].
  pose proof (step_result_free_of_arrays s Hs) as Hresult.
  destruct (step s) as [next | k vs hp rd | ]; [ exact (IH next Hresult) | .. ];
    exact Hresult.
Qed.

Theorem run_free_of_arrays : run_free_of_arrays_statement.
Proof.
  intros n s Hs.
  exact
    (result_free_of_arrays_statement_form _ (run_result_free_of_arrays n s Hs)).
Qed.

(* ======================================================================== *)
(* The machine with a heap                                                  *)
(* ======================================================================== *)

Definition heap_result_free_of_arrays (result : heap_step_result) : Prop :=
  match result with
  | Heap_step s' => heap_state_free_of_arrays s'
  | Heap_exit _ o => outcome_no_array_value o
  | Heap_undefined => True
  end.

(* A [Let] of a primitive is run by [exec_prim]; any other step is the
   step of the machine, with the heap carried along. *)
Lemma heap_step_result_free_of_arrays (s : heap_state) :
  heap_state_free_of_arrays s -> heap_result_free_of_arrays (heap_step s).
Proof.
  intros Hs. destruct (is_let_prim (hs_expr s)) eqn:Hprim.
  - destruct s as [e vars conts statics h]. simpl in Hprim.
    destruct e as [x kind d body | | | | | | | ]; try discriminate.
    destruct d as [ | p args | | | | | ]; try discriminate.
    destruct Hs as (Hexpr & Hvars & Hconts & Hcode & Hstatics & Hheap).
    simpl in Hexpr, Hvars, Hconts, Hcode, Hstatics, Hheap.
    apply orb_false_iff in Hexpr as [Hp Hbody].
    unfold heap_step, heap_exec_named.
    destruct (eval_simples vars args) as [vs | ] eqn:Hargs; simpl;
      [ | exact I ].
    destruct (exec_prim statics h p vs) as [[v h'] | ] eqn:Hexec;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    destruct
      (exec_prim_no_array_value statics h h' p vs v Hp Hstatics Hheap
         (proj1 (forallb_forall _ _)
            (eval_simples_no_array_value vars Hvars args vs Hargs))
         Hexec)
      as [Hv Hh'].
    repeat split; simpl; try assumption.
    intros y w [[= <- <-] | Hin]; [ exact Hv | exact (Hvars y w Hin) ].
  - pose proof (heap_step_agrees s Hprim) as Hagrees.
    pose proof (step_result_free_of_arrays (forget_heap s) Hs) as Hresult.
    destruct Hs as (_ & _ & _ & _ & Hstatics & Hheap).
    destruct (heap_step s) as [s' | k o | ],
      (step (forget_heap s)) as [t | k' vs hp rd | ];
      try contradiction.
    + destruct Hagrees as [<- _]. exact Hresult.
    + destruct Hagrees as [_ ->]. destruct Hresult as [Hvs _].
      split; [ | split ].
      * apply forallb_forall. exact Hvs.
      * exact Hstatics.
      * exact Hheap.
    + exact I.
Qed.

Theorem heap_step_free_of_arrays : heap_step_free_of_arrays_statement.
Proof. exact heap_step_result_free_of_arrays. Qed.

Theorem heap_run_free_of_arrays : heap_run_free_of_arrays_statement.
Proof.
  intros n. induction n as [ | n IH]; intros s Hs; simpl; [ exact Hs | ].
  pose proof (heap_step_result_free_of_arrays s Hs) as Hresult.
  destruct (heap_step s) as [next | k o | ]; [ exact (IH next Hresult) | .. ];
    exact Hresult.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The law of [makes_array] is proved from the definitions by the
   tactic of FlambdaEquivarianceTactic.v ([equivariance]), which
   states and proves by itself the law of [named_makes_array]. The
   law is given back to the tactic for those of the handler and of
   the code, which are of the same recursion. *)
(* The step of the database is the one pass: every proof by the tactic
   of this part passes with it. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

Lemma rename_makes_array (r : renaming) (e : expr) :
  makes_array (rename_expr r e) = makes_array e.
Proof. equivariance. Qed.
#[local] Hint Rewrite rename_makes_array : equivariance.

Theorem makes_array_renamed : makes_array_renamed_statement.
Proof. exact rename_makes_array. Qed.

Lemma rename_handler_makes_array (r : renaming) (h : continuation_handler) :
  handler_makes_array (rename_handler r h) = handler_makes_array h.
Proof. equivariance. Qed.

Lemma rename_code_makes_array (r : renaming) (c : code) :
  code_makes_array (rename_code r c) = code_makes_array c.
Proof. equivariance. Qed.

Lemma rename_env_no_array_value (r : renaming) (vars : env) :
  env_no_array_value vars -> env_no_array_value (rename_env r vars).
Proof.
  intros Hvars x v Hin. unfold rename_env in Hin.
  apply in_map_iff in Hin as ([x' v'] & Heq & Hin).
  injection Heq as <- <-. exact (Hvars x' v' Hin).
Qed.

Lemma rename_cont_free_of_arrays (r : renaming) (c : cont_value) :
  cont_free_of_arrays c -> cont_free_of_arrays (rename_cont_value r c).
Proof.
  induction 1
    as [k | c extra Hc IH Hextra
       | h vars conts recursive Hh Hvars Hconts IHconts Hrecursive
       | remaining remaining_arity return_arity return_to exn_to Hremaining
           Hreturn IHreturn Hexn IHexn];
    simpl.
  - constructor.
  - constructor; assumption.
  - constructor.
    + rewrite rename_handler_makes_array. exact Hh.
    + exact (rename_env_no_array_value r vars Hvars).
    + intros k c Hin.
      apply in_map_iff in Hin as ([k' c'] & Heq & Hin).
      injection Heq as <- <-. exact (IHconts k' c' Hin).
    + intros k h' Hin. unfold rename_handlers in Hin.
      apply in_map_iff in Hin as ([k' h0] & Heq & Hin).
      injection Heq as <- <-. rewrite rename_handler_makes_array.
      exact (Hrecursive k' h0 Hin).
  - constructor; assumption.
Qed.

Lemma rename_state_free_of_arrays (r : renaming) (s : state) :
  state_free_of_arrays s -> state_free_of_arrays (rename_state r s).
Proof.
  intros (Hexpr & Hvars & Hconts & Hcode & Hstatics & Hheap).
  split; [ | split; [ | split; [ | split; [ | split ] ] ] ]; simpl.
  - rewrite rename_makes_array. exact Hexpr.
  - exact (rename_env_no_array_value r _ Hvars).
  - intros k c Hin. unfold rename_cont_env in Hin.
    apply in_map_iff in Hin as ([k' c'] & Heq & Hin).
    injection Heq as <- <-.
    exact (rename_cont_free_of_arrays r c' (Hconts k' c' Hin)).
  - intros id c Hin. unfold map_values in Hin.
    apply in_map_iff in Hin as ([id' c'] & Heq & Hin).
    injection Heq as <- <-. rewrite rename_code_makes_array.
    exact (Hcode id' c' Hin).
  - exact Hstatics.
  - exact Hheap.
Qed.

Theorem state_free_of_arrays_renamed :
  state_free_of_arrays_renamed_statement.
Proof.
  intros r s. split; [ | apply rename_state_free_of_arrays ].
  intros Hs.
  rewrite <- (proj1 (rename_state_inverse r s)).
  exact (rename_state_free_of_arrays (inverse_renaming r) _ Hs).
Qed.

Theorem heap_state_free_of_arrays_renamed :
  heap_state_free_of_arrays_renamed_statement.
Proof.
  intros r s. unfold heap_state_free_of_arrays.
  rewrite forget_heap_equivariant. apply state_free_of_arrays_renamed.
Qed.

(* By the proof of
   FlambdaHeapEquivarianceProofs.heap_expr_refines_equivariant, with
   the second condition on the state carried along. *)
Theorem heap_expr_refines_free_of_arrays_equivariant :
  heap_expr_refines_free_of_arrays_equivariant_statement.
Proof.
  intros r source target Hrefines vars conts statics h Hclosed Hfree.
  rewrite !heap_state_of_renamed_expr.
  rewrite heap_state_of_renamed_expr in Hclosed, Hfree.
  apply heap_refines_equivariant, Hrefines.
  - apply (state_closed_renamed (inverse_renaming r)) in Hclosed.
    destruct (rename_heap_state_inverse r
                (Heap_state source (rename_env (inverse_renaming r) vars)
                   (rename_cont_env (inverse_renaming r) conts)
                   (rename_static_env (inverse_renaming r) statics) h))
      as [Hleft _].
    rewrite Hleft in Hclosed. exact Hclosed.
  - exact (proj1 (heap_state_free_of_arrays_renamed r _) Hfree).
Qed.

(* The witness of FlambdaHeapMachineProofs.delete_write_wrong, a write
   into a block, and its proof, with the state shown free of arrays. *)
Theorem delete_write_wrong_free_of_arrays :
  delete_write_wrong_free_of_arrays_statement.
Proof.
  pose (kind := Block_access_values None None Any_value).
  pose (body :=
          Apply_cont (Apply_cont_expr 3%positive [Var 2%positive] None)).
  pose (one := Value_const (Tagged_immediate 1)).
  pose (five := Value_const (Tagged_immediate 5)).
  pose (h := [(1%positive, Cell_block Cell_mutable 0 [one])]).
  exists 1%positive, Value, (Heap (Block_set kind Assignment 0)),
    [Var 2%positive; Const (Tagged_immediate 5)], body.
  split; [ reflexivity | ].
  intros Hrefines.
  specialize
    (Hrefines [(2%positive, Value_location 1%positive)] [] empty_static_env h).
  destruct Hrefines as [Hexits _].
  { split; [ split | split ]; simpl.
    - intros l c v. unfold h.
      destruct l; simpl; try discriminate.
      intros [= <-]. simpl. intros [<- | [ ]].
      split; [ intros l [ ] | reflexivity ].
    - intros s a v. discriminate.
    - intros y v [[= <- <-] | [ ]]. intros l [<- | [ ]]. reflexivity.
    - intros k c [ ]. }
  { split; [ reflexivity | ]. simpl.
    split; [ | split; [ | split; [ | split ] ] ].
    - intros y v [[= <- <-] | [ ]]. reflexivity.
    - intros k c [ ].
    - intros id c [ ].
    - intros s a [ ].
    - intros l c [[= <- <-] | [ ]]. reflexivity. }
  destruct
    (Hexits 3%positive
       (Outcome [Value_location 1%positive] empty_static_env
          ((1%positive, Cell_block Cell_mutable 0 [five]) :: h)))
    as (o' & [fuel Htarget] & Hrelated).
  { exists 2%nat. reflexivity. }
  destruct fuel as [ | fuel]; [ discriminate | ].
  simpl in Htarget. injection Htarget as <-.
  destruct Hrelated as (r & Hvalid & Hvalues).
  simpl in Hvalid, Hvalues.
  inversion Hvalues as [ | v v' vs vs' Hfirst _]; subst.
  apply (value_related_location _ _ _ _ _ _ _ _ Hvalid) in Hfirst
    as (l' & [= <-] & Hl).
  destruct (proj1 (proj2 Hvalid) _ _ Hl)
    as (c & c' & [= <-] & [= <-] & Hcell).
  destruct Hcell as (_ & _ & Hfields).
  inversion Hfields as [ | field field' fields fields' Hfield _]; subst.
  apply (value_related_immediate _ _ _ _ _ _ _ _ Hvalid) in Hfield.
  discriminate.
Qed.

(* ======================================================================== *)
(* The relation of the machine with a heap                                  *)
(* ======================================================================== *)

Lemma Forall2_no_array_value (R : value -> value -> Prop)
    (l l' : list value) :
  Forall2 R l l' ->
  forallb no_array_value l = true ->
  forallb no_array_value l' = true ->
  Forall2
    (fun w w' => R w w' ∧ no_array_value w = true ∧ no_array_value w' = true)
    l l'.
Proof.
  induction 1 as [ | w w' l l' Hw _ IH]; simpl; [ constructor | ].
  intros Hl Hl'. apply andb_true_iff in Hl as [Hw1 Hl].
  apply andb_true_iff in Hl' as [Hw1' Hl'].
  constructor; [ auto | exact (IH Hl Hl') ].
Qed.

(* A rule is the same rule under a correspondence that relates the same
   locations. *)
Lemma value_step_same_locations (poison_refines : bool)
    (r r' : correspondence) (statics statics' : static_env)
    (R : value -> value -> Prop) (v v' : value) :
  (∀ l l' : location, corr_locations r l l' -> corr_locations r' l l') ->
  value_step poison_refines r statics statics' R v v' ->
  value_step poison_refines r' statics statics' R v v'.
Proof.
  intros Hlocations H.
  destruct H
    as [c | l l' Hl | v v' tag fields fields' Hb Hb' Hfields
       | v v' c c' Hn Hn' Hc | c | v v' bytes Hs Hs'
       | v v' fslot fdecls slots slots' Hc Hc' Hnames Hslots Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Ha Ha' Helements].
  - apply Step_const.
  - apply Step_location. exact (Hlocations l l' Hl).
  - exact (Step_block _ _ _ _ _ v v' tag fields fields' Hb Hb' Hfields).
  - exact (Step_boxed_number _ _ _ _ _ v v' c c' Hn Hn' Hc).
  - apply Step_same_boxed_number.
  - exact (Step_string _ _ _ _ _ v v' bytes Hs Hs').
  - exact
      (Step_closure _ _ _ _ _ v v' fslot fdecls slots slots' Hc Hc' Hnames
         Hslots Hsiblings).
  - exact (Step_same_closure_of_set _ _ _ _ _ gslot gdecls gslots Hgslot).
  - exact (Step_opaque_symbol _ _ _ _ _ s Hs Hs').
  - exact (Step_poison _ _ _ _ _ kind v' Hpoison Hkind).
  - exact
      (Step_partial_application _ _ _ _ _ pcallee pcallee' pargs pargs'
         parity Hpcallee Hpargs).
  - exact (Step_array _ _ _ _ _ v v' elements elements' Ha Ha' Helements).
Qed.

Section Relation.

Variable poison_refines : bool.
Variable r : correspondence.
Variables statics statics' : static_env.
Hypothesis Hstatics : statics_no_array_value statics.
Hypothesis Hstatics' : statics_no_array_value statics'.

Lemma value_step_no_array_value_here (R : value -> value -> Prop)
    (v v' : value) :
  no_array_value v = true ->
  no_array_value v' = true ->
  value_step poison_refines r statics statics' R v v' ->
  value_step poison_refines r statics statics'
    (fun w w' => R w w' ∧ no_array_value w = true ∧ no_array_value w' = true)
    v v'.
Proof.
  intros Hv Hv' H. revert Hv Hv'.
  destruct H
    as [c | l l' Hl | v v' tag fields fields' Hb Hb' Hfields
       | v v' c c' Hn Hn' Hc | c | v v' bytes Hs Hs'
       | v v' fslot fdecls slots slots' Hc Hc' Hnames Hslots Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Ha Ha' Helements];
    intros Hv Hv'.
  - apply Step_const.
  - apply Step_location. exact Hl.
  - apply (Step_block _ _ _ _ _ v v' tag fields fields' Hb Hb').
    exact
      (Forall2_no_array_value R fields fields' Hfields
         (block_of_no_array_value statics Hstatics v tag fields Hv Hb)
         (block_of_no_array_value statics' Hstatics' v' tag fields' Hv' Hb')).
  - exact (Step_boxed_number _ _ _ _ _ v v' c c' Hn Hn' Hc).
  - apply Step_same_boxed_number.
  - exact (Step_string _ _ _ _ _ v v' bytes Hs Hs').
  - pose proof
      (closure_in_set_no_array_value statics Hstatics v fslot fdecls slots
         Hv Hc)
      as Hfree.
    pose proof
      (closure_in_set_no_array_value statics' Hstatics' v' fslot fdecls
         slots' Hv' Hc')
      as Hfree'.
    unfold slots_no_array_value in Hfree, Hfree'.
    rewrite <- forallb_of_map in Hfree, Hfree'.
    apply (Step_closure _ _ _ _ _ v v' fslot fdecls slots slots' Hc Hc'
             Hnames);
      [ exact (Forall2_no_array_value R _ _ Hslots Hfree Hfree') | ].
    destruct Hsiblings as [Hbound | [Hsome Hnone]];
      [ left; exact Hbound | right; split; [ | exact Hnone ] ].
    intros move_to w Hw. destruct (Hsome move_to w Hw) as (w' & Hw' & HR).
    exists w'. split; [ exact Hw' | ]. split; [ exact HR | ]. split.
    + exact
        (project_function_slot_no_array_value statics fslot move_to v w Hv
           Hw).
    + exact
        (project_function_slot_no_array_value statics' fslot move_to v' w'
           Hv' Hw').
  - exact (Step_same_closure_of_set _ _ _ _ _ gslot gdecls gslots Hgslot).
  - exact (Step_opaque_symbol _ _ _ _ _ s Hs Hs').
  - exact (Step_poison _ _ _ _ _ kind v' Hpoison Hkind).
  - simpl in Hv, Hv'. apply andb_true_iff in Hv as [Hcallee Hargs].
    apply andb_true_iff in Hv' as [Hcallee' Hargs'].
    apply Step_partial_application; [ auto | ].
    exact (Forall2_no_array_value R pargs pargs' Hpargs Hargs Hargs').
  - apply (Step_array _ _ _ _ _ v v' elements elements' Ha Ha').
    exact
      (Forall2_no_array_value R elements elements' Helements
         (array_of_elements_no_array_value statics Hstatics v elements Hv Ha)
         (array_of_elements_no_array_value statics' Hstatics' v' elements'
            Hv' Ha')).
Qed.

Let restricted : correspondence := restrict_to_no_array_value r.

Lemma value_related_restrict (v v' : value) :
  value_related poison_refines r statics statics' v v' ->
  no_array_value v = true ->
  no_array_value v' = true ->
  value_related poison_refines restricted statics statics' v v'.
Proof.
  revert v v'.
  apply
    (value_related_induction poison_refines r statics statics'
       (fun v v' =>
          no_array_value v = true ->
          no_array_value v' = true ->
          value_related poison_refines restricted statics statics' v v')).
  - intros v v' Hcorr Hv Hv'. apply Related_corr. simpl. auto.
  - intros v v' Hstep Hv Hv'. apply Related_step.
    apply (value_step_same_locations poison_refines r restricted);
      [ intros l l' Hl; exact Hl | ].
    apply (value_step_no_array_value_here _ v v' Hv Hv') in Hstep.
    revert Hstep. apply value_step_mono.
    intros a a' [[_ IH] [Ha Ha']]. exact (IH Ha Ha').
Qed.

Lemma value_related_unrestrict (v v' : value) :
  value_related poison_refines restricted statics statics' v v' ->
  value_related poison_refines r statics statics' v v'.
Proof.
  revert v v'.
  apply
    (value_related_induction poison_refines restricted statics statics'
       (fun v v' => value_related poison_refines r statics statics' v v')).
  - intros v v' [Hcorr _]. apply Related_corr. exact Hcorr.
  - intros v v' Hstep. apply Related_step.
    apply (value_step_same_locations poison_refines restricted r);
      [ intros l l' Hl; exact Hl | ].
    revert Hstep. apply value_step_mono. intros a a' [_ IH]. exact IH.
Qed.

Lemma values_related_restrict (vs vs' : list value) :
  Forall2 (value_related poison_refines r statics statics') vs vs' ->
  forallb no_array_value vs = true ->
  forallb no_array_value vs' = true ->
  Forall2 (value_related poison_refines restricted statics statics') vs vs'.
Proof.
  intros Hrelated Hvs Hvs'.
  apply (Forall2_no_array_value _ _ _ Hrelated Hvs) in Hvs'.
  revert Hvs'. apply Forall2_mono.
  intros a a' (Ha & Hfree & Hfree').
  exact (value_related_restrict a a' Ha Hfree Hfree').
Qed.

Lemma correspondence_valid_restrict (h h' : heap) :
  heap_no_array_value h ->
  heap_no_array_value h' ->
  correspondence_valid poison_refines r statics statics' h h' ->
  correspondence_valid poison_refines restricted statics statics' h h'.
Proof.
  intros Hheap Hheap' (Hbijection & Hcells & Hvalues).
  split; [ exact Hbijection | split ].
  - intros l l' Hl.
    destruct (Hcells l l' Hl) as (c & c' & Hc & Hc' & Hrelated).
    exists c, c'. split; [ exact Hc | split; [ exact Hc' | ] ].
    pose proof (Hheap l c (lookup_in _ _ _ Hc)) as Hfree.
    pose proof (Hheap' l' c' (lookup_in _ _ _ Hc')) as Hfree'.
    unfold cell_no_array_value in Hfree, Hfree'.
    destruct c as [m t fs | k es], c' as [m' t' fs' | k' es'];
      simpl in Hrelated, Hfree, Hfree' |- *; try contradiction.
    + destruct Hrelated as (Hm & Ht & Hfs).
      split; [ exact Hm | split; [ exact Ht | ] ].
      exact (values_related_restrict fs fs' Hfs Hfree Hfree').
    + destruct Hrelated as (Hk & Hes). split; [ exact Hk | ].
      exact (values_related_restrict es es' Hes Hfree Hfree').
  - intros v v' (Hcorr & Hv & Hv').
    apply (value_step_same_locations poison_refines r restricted);
      [ intros l l' Hl; exact Hl | ].
    pose proof (Hvalues v v' Hcorr) as Hstep.
    apply (value_step_no_array_value_here _ v v' Hv Hv') in Hstep.
    revert Hstep. apply value_step_mono.
    intros a a' (Ha & Hfree & Hfree').
    exact (value_related_restrict a a' Ha Hfree Hfree').
Qed.

End Relation.

Theorem value_step_no_array_value : value_step_no_array_value_statement.
Proof.
  intros poison_refines r statics statics' R v v' Hstatics Hstatics'.
  exact
    (value_step_no_array_value_here poison_refines r statics statics'
       Hstatics Hstatics' R v v').
Qed.

Theorem value_related_restricted : value_related_restricted_statement.
Proof.
  intros poison_refines r statics statics' h h' Hstatics Hstatics' Hheap
    Hheap' Hvalid.
  split.
  - exact
      (correspondence_valid_restrict poison_refines r statics statics'
         Hstatics Hstatics' h h' Hheap Hheap' Hvalid).
  - intros v v' Hv Hv'. split.
    + intros Hrelated.
      exact
        (value_related_restrict poison_refines r statics statics' Hstatics
           Hstatics' v v' Hrelated Hv Hv').
    + exact (value_related_unrestrict poison_refines r statics statics' v v').
Qed.

Theorem outcome_related_no_array_value :
  outcome_related_no_array_value_statement.
Proof.
  intros poison_refines o o' (Hvs & Hstatics & Hheap)
    (Hvs' & Hstatics' & Hheap').
  split.
  - intros (r & Hvalid & Hvalues).
    exists (restrict_to_no_array_value r). split; [ | split ].
    + intros v v' (_ & Hv & Hv'). split; assumption.
    + exact
        (correspondence_valid_restrict poison_refines r _ _ Hstatics
           Hstatics' _ _ Hheap Hheap' Hvalid).
    + apply
        (values_related_restrict poison_refines r _ _ Hstatics Hstatics' _ _
           Hvalues);
        apply forallb_forall; assumption.
  - intros (r & _ & Hvalid & Hvalues). exists r. split; assumption.
Qed.

(* ======================================================================== *)
(* The relation of the lifting                                              *)
(* ======================================================================== *)

Theorem value_lifted_array : value_lifted_array_statement.
Proof.
  intros statics' elements v' H. inversion H; subst.
  - eexists. split; [ eassumption | left; reflexivity ].
  - eexists. split; [ eassumption | right ].
    eexists. split; [ reflexivity | eassumption ].
Qed.

Theorem value_lifted_to_array : value_lifted_to_array_statement.
Proof.
  intros statics' v elements' H. inversion H; subst.
  eexists. split; [ reflexivity | assumption ].
Qed.

Fixpoint value_lifted_keeps_no_array_value (statics' : static_env)
    (v v' : value) (H : value_lifted statics' v v') {struct H} :
  no_array_value v = true -> no_array_value v' = true.
Proof.
  destruct H
    as [ c | s | c | tag fields fields' Hfields | items items' Hitems
       | code slots slots' Hslots | fslot decls slots slots' Hslots
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | c s Hlookup | tag fields fields' s Hlookup Hfields
       | items items' s Hlookup Hitems
       | fslot decls slots slots' s syms Hsym Hsyms Hbound Hslots
       | code slots slots' s Hlookup Hslots | l ];
    simpl; try (intros _; reflexivity).
  - revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; simpl;
      [ reflexivity | ].
    intros Hall. apply andb_true_iff in Hall as [Hw Hrest].
    rewrite (value_lifted_keeps_no_array_value _ _ _ Hhead Hw).
    exact (IH _ _ Htail Hrest).
  - discriminate.
  - revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      [ reflexivity | ].
    intros Hall. apply andb_true_iff in Hall as [Hw Hrest].
    rewrite (value_lifted_keeps_no_array_value _ _ _ Hhead Hw).
    exact (IH _ _ Htail Hrest).
  - revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      [ reflexivity | ].
    intros Hall. apply andb_true_iff in Hall as [Hw Hrest].
    rewrite (value_lifted_keeps_no_array_value _ _ _ Hhead Hw).
    exact (IH _ _ Htail Hrest).
  - intros Hall. apply andb_true_iff in Hall as [Hcallee Hargs].
    rewrite (value_lifted_keeps_no_array_value _ _ _ Hpcallee Hcallee).
    simpl. revert pargs pargs' Hpargs Hargs. fix IH 3.
    intros pargs pargs' [ | w w' rest rest' Hhead Htail]; simpl;
      [ reflexivity | ].
    intros Hall. apply andb_true_iff in Hall as [Hw Hrest].
    rewrite (value_lifted_keeps_no_array_value _ _ _ Hhead Hw).
    exact (IH _ _ Htail Hrest).
Qed.

Theorem value_lifted_no_array_value : value_lifted_no_array_value_statement.
Proof. exact value_lifted_keeps_no_array_value. Qed.

(* A binding that a filter keeps is found as it was. *)
Lemma lookup_filter_kept {A : Type} (keep : positive * A -> bool)
    (name : positive) (bindings : list (positive * A)) (a : A) :
  lookup name bindings = Some a ->
  keep (name, a) = true ->
  lookup name (filter keep bindings) = Some a.
Proof.
  induction bindings as [ | [bound b] bindings IH]; simpl;
    [ discriminate | ].
  destruct (Pos.eqb_spec name bound) as [<- | Hne].
  - intros [= ->] Hkeep. rewrite Hkeep. simpl. rewrite Pos.eqb_refl.
    reflexivity.
  - intros Hlookup Hkeep. destruct (keep (bound, b)); simpl.
    + destruct (Pos.eqb_spec name bound) as [Heq | _];
        [ contradiction | exact (IH Hlookup Hkeep) ].
    + exact (IH Hlookup Hkeep).
Qed.

(* And a binding found in what a filter keeps is one that it keeps. *)
Lemma lookup_filter_keeps {A : Type} (keep : positive * A -> bool)
    (name : positive) (bindings : list (positive * A)) (a : A) :
  lookup name (filter keep bindings) = Some a -> keep (name, a) = true.
Proof.
  induction bindings as [ | [bound b] bindings IH]; simpl;
    [ discriminate | ].
  destruct (keep (bound, b)) eqn:Hkeep; simpl; [ | exact IH ].
  destruct (Pos.eqb_spec name bound) as [-> | _]; [ | exact IH ].
  intros [= <-]. exact Hkeep.
Qed.

Lemma lookup_without_static_arrays (statics : static_env) (s : symbol)
    (a : statically_allocated_value) :
  lookup s (static_values statics) = Some a ->
  is_static_array a = false ->
  lookup s (static_values (without_static_arrays statics)) = Some a.
Proof.
  intros Hlookup Hnot. unfold without_static_arrays. simpl.
  apply lookup_filter_kept; [ exact Hlookup | ]. simpl. rewrite Hnot.
  reflexivity.
Qed.

Theorem no_array_lifted_to_symbol : no_array_lifted_to_symbol_statement.
Proof.
  intros statics' elements s H. inversion H; subst.
  match goal with
  | Hlookup : lookup s _ = Some (Static_immutable_array _) |- _ =>
    apply lookup_filter_keeps in Hlookup; discriminate Hlookup
  end.
Qed.

Fixpoint value_lifted_without_static_arrays (statics' : static_env)
    (v v' : value) (H : value_lifted statics' v v') {struct H} :
  no_array_value v = true ->
  value_lifted (without_static_arrays statics') v v'.
Proof.
  destruct H
    as [ c | s | c | tag fields fields' Hfields | items items' Hitems
       | code slots slots' Hslots | fslot decls slots slots' Hslots
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | c s Hlookup | tag fields fields' s Hlookup Hfields
       | items items' s Hlookup Hitems
       | fslot decls slots slots' s syms Hsym Hsyms Hbound Hslots
       | code slots slots' s Hlookup Hslots | l ];
    simpl; intros Hno; try discriminate Hno.
  - apply Lifted_const.
  - apply Lifted_symbol.
  - apply Lifted_boxed_number.
  - apply Lifted_block.
    revert fields fields' Hfields Hno. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + exact (value_lifted_without_static_arrays _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply Lifted_closure.
    revert slots slots' Hslots Hno. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + split; [ exact Hkey | ].
      exact (value_lifted_without_static_arrays _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply Lifted_closure_of_set.
    revert slots slots' Hslots Hno. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + split; [ exact Hkey | ].
      exact (value_lifted_without_static_arrays _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply andb_true_iff in Hno as [Hcallee Hargs].
    apply Lifted_partial_application;
      [ exact (value_lifted_without_static_arrays _ _ _ Hpcallee Hcallee)
      | ].
    revert pargs pargs' Hpargs Hargs. fix IH 3.
    intros pargs pargs' [ | w w' rest rest' Hhead Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + exact (value_lifted_without_static_arrays _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply Lifted_to_boxed_number.
    exact (lookup_without_static_arrays _ _ _ Hlookup eq_refl).
  - apply (Lifted_to_block _ tag fields fields' s);
      [ exact (lookup_without_static_arrays _ _ _ Hlookup eq_refl) | ].
    clear Hlookup. revert fields fields' Hfields Hno. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + exact (value_lifted_without_static_arrays _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply (Lifted_to_closure _ fslot decls slots slots' s syms Hsym Hsyms).
    + intros slot sym Hin.
      exact (lookup_without_static_arrays _ _ _ (Hbound slot sym Hin)
               eq_refl).
    + clear Hbound. revert slots slots' Hslots Hno. fix IH 3.
      intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
        intros Hall; constructor;
        apply andb_true_iff in Hall as [Hw Hrest].
      * split; [ exact Hkey | ].
        exact (value_lifted_without_static_arrays _ _ _ Hhead Hw).
      * exact (IH _ _ Htail Hrest).
  - apply (Lifted_to_unnamed_closure _ code slots slots' s);
      [ exact (lookup_without_static_arrays _ _ _ Hlookup eq_refl) | ].
    clear Hlookup. revert slots slots' Hslots Hno. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + split; [ exact Hkey | ].
      exact (value_lifted_without_static_arrays _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply Lifted_location.
Qed.

Theorem value_lifted_reads_no_static_array :
  value_lifted_reads_no_static_array_statement.
Proof. exact value_lifted_without_static_arrays. Qed.

(* A unit run on its own starts in a state free of arrays, if it makes
   none. *)
Lemma unit_state_free_of_arrays (e : expr) :
  makes_array e = false ->
  state_free_of_arrays (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hmakes. unfold state_free_of_arrays. simpl.
  split; [ exact Hmakes | ].
  repeat split; intros ? ? [].
Qed.

(* Two heaps alike, the first of which holds no array value, are alike
   up to a relation that relates what the first relates of a value
   that holds none. *)
Lemma heap_alike_no_array (R R' : value -> value -> Prop) (h h' : heap) :
  (∀ v v' : value, R v v' -> no_array_value v = true -> R' v v') ->
  heap_no_array_value h ->
  heap_alike R h h' ->
  heap_alike R' h h'.
Proof.
  intros Himpl Hnone Halike.
  assert (Hvalues :
    ∀ vs vs' : list value,
      forallb no_array_value vs = true ->
      Forall2 R vs vs' ->
      Forall2 R' vs vs').
  { intros vs vs' Hvs Hrel. rewrite forallb_forall in Hvs.
    induction Hrel as [ | v v' vs vs' Hv _ IH]; constructor.
    - exact (Himpl v v' Hv (Hvs v (or_introl eq_refl))).
    - apply IH. intros w Hw. exact (Hvs w (or_intror Hw)). }
  induction Halike as [ | [l c] [l' c'] h h' [Hkey Hcell] _ IH]; constructor.
  - split; [ exact Hkey | ]. simpl in Hcell |- *.
    pose proof (Hnone l c (or_introl eq_refl)) as Hc.
    unfold cell_no_array_value in Hc.
    destruct c as [mut tag fields | kind elements],
      c' as [mut' tag' fields' | kind' elements'];
      try contradiction; simpl in Hc.
    + destruct Hcell as (Hmut & Htag & Hfields).
      split; [ exact Hmut | split; [ exact Htag | ] ].
      exact (Hvalues fields fields' Hc Hfields).
    + destruct Hcell as (Hkind & Helements).
      split; [ exact Hkind | exact (Hvalues elements elements' Hc Helements) ].
  - apply IH. intros l0 c0 Hin. exact (Hnone l0 c0 (or_intror Hin)).
Qed.

Theorem lift_simulates_free_of_arrays :
  lift_simulates_free_of_arrays_statement.
Proof.
  intros u lets known e. cbv zeta. intros Hwf Honly Hmakes k vs hp Hexit.
  assert (Hfree :
    (∀ v : value, v ∈ vs -> no_array_value v = true)
    ∧ heap_no_array_value hp).
  { destruct Hexit as [fuel [rd Hrun]].
    pose proof
      (run_free_of_arrays fuel _ (unit_state_free_of_arrays e Hmakes))
      as Hfree.
    rewrite Hrun in Hfree. exact Hfree. }
  destruct Hfree as [Hvs Hhp].
  split; [ exact Hvs | ]. split; [ exact Hhp | ].
  pose proof (lift_simulates u lets known e) as Hsimulates.
  cbv zeta in Hsimulates.
  destruct (proj1 (Hsimulates Hwf Honly) k vs hp Hexit)
    as (n & last & vs' & hp' & rd & Hrun & Hstep & Hlifted & Hheap).
  exists n, last, vs', hp', rd.
  split; [ exact Hrun | split; [ exact Hstep | ] ].
  split.
  - clear Hrun Hstep Hexit Hheap.
    induction Hlifted as [ | v v' vs vs' Hv _ IH]; constructor.
    + exact (value_lifted_reads_no_static_array _ _ _ Hv
               (Hvs v (or_introl eq_refl))).
    + apply IH. intros w Hw. exact (Hvs w (or_intror Hw)).
  - exact (heap_alike_no_array _ _ hp hp'
             (value_lifted_reads_no_static_array _) Hhp Hheap).
Qed.

(* ======================================================================== *)
(* The rules of the lifting that have no array                              *)
(* ======================================================================== *)

(* Each is a rule of the relation. *)
Fixpoint value_lifted_no_array_lifted (statics' : static_env)
    (v v' : value) (H : value_lifted_no_array statics' v v') {struct H} :
  value_lifted statics' v v'.
Proof.
  destruct H
    as [ c | s | c | tag fields fields' Hfields
       | code slots slots' Hslots | fslot decls slots slots' Hslots
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | c s Hlookup | tag fields fields' s Hlookup Hfields
       | fslot decls slots slots' s syms Hsym Hsyms Hbound Hslots
       | code slots slots' s Hlookup Hslots | l ].
  - apply Lifted_const.
  - apply Lifted_symbol.
  - apply Lifted_boxed_number.
  - apply Lifted_block.
    revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; constructor.
    + exact (value_lifted_no_array_lifted _ _ _ Hhead).
    + exact (IH _ _ Htail).
  - apply Lifted_closure.
    revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail];
      constructor.
    + split; [ exact Hkey | ].
      exact (value_lifted_no_array_lifted _ _ _ Hhead).
    + exact (IH _ _ Htail).
  - apply Lifted_closure_of_set.
    revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail];
      constructor.
    + split; [ exact Hkey | ].
      exact (value_lifted_no_array_lifted _ _ _ Hhead).
    + exact (IH _ _ Htail).
  - apply Lifted_partial_application;
      [ exact (value_lifted_no_array_lifted _ _ _ Hpcallee) | ].
    revert pargs pargs' Hpargs. fix IH 3.
    intros pargs pargs' [ | w w' rest rest' Hhead Htail]; constructor.
    + exact (value_lifted_no_array_lifted _ _ _ Hhead).
    + exact (IH _ _ Htail).
  - apply Lifted_to_boxed_number. exact Hlookup.
  - apply (Lifted_to_block _ tag fields fields' s Hlookup).
    clear Hlookup. revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; constructor.
    + exact (value_lifted_no_array_lifted _ _ _ Hhead).
    + exact (IH _ _ Htail).
  - apply (Lifted_to_closure _ fslot decls slots slots' s syms Hsym Hsyms
             Hbound).
    clear Hbound. revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail];
      constructor.
    + split; [ exact Hkey | ].
      exact (value_lifted_no_array_lifted _ _ _ Hhead).
    + exact (IH _ _ Htail).
  - apply (Lifted_to_unnamed_closure _ code slots slots' s Hlookup).
    clear Hlookup. revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail];
      constructor.
    + split; [ exact Hkey | ].
      exact (value_lifted_no_array_lifted _ _ _ Hhead).
    + exact (IH _ _ Htail).
  - apply Lifted_location.
Qed.

(* What they relate has a first value that holds no array value. *)
Fixpoint value_lifted_no_array_holds_none (statics' : static_env)
    (v v' : value) (H : value_lifted_no_array statics' v v') {struct H} :
  no_array_value v = true.
Proof.
  destruct H
    as [ c | s | c | tag fields fields' Hfields
       | code slots slots' Hslots | fslot decls slots slots' Hslots
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | c s Hlookup | tag fields fields' s Hlookup Hfields
       | fslot decls slots slots' s syms Hsym Hsyms Hbound Hslots
       | code slots slots' s Hlookup Hslots | l ];
    simpl; try reflexivity.
  - revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; simpl;
      [ reflexivity | ].
    rewrite (value_lifted_no_array_holds_none _ _ _ Hhead).
    exact (IH _ _ Htail).
  - revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      [ reflexivity | ].
    rewrite (value_lifted_no_array_holds_none _ _ _ Hhead).
    exact (IH _ _ Htail).
  - revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      [ reflexivity | ].
    rewrite (value_lifted_no_array_holds_none _ _ _ Hhead).
    exact (IH _ _ Htail).
  - rewrite (value_lifted_no_array_holds_none _ _ _ Hpcallee). simpl.
    revert pargs pargs' Hpargs. fix IH 3.
    intros pargs pargs' [ | w w' rest rest' Hhead Htail]; simpl;
      [ reflexivity | ].
    rewrite (value_lifted_no_array_holds_none _ _ _ Hhead).
    exact (IH _ _ Htail).
  - clear Hlookup. revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; simpl;
      [ reflexivity | ].
    rewrite (value_lifted_no_array_holds_none _ _ _ Hhead).
    exact (IH _ _ Htail).
  - clear Hbound. revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      [ reflexivity | ].
    rewrite (value_lifted_no_array_holds_none _ _ _ Hhead).
    exact (IH _ _ Htail).
  - clear Hlookup. revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      [ reflexivity | ].
    rewrite (value_lifted_no_array_holds_none _ _ _ Hhead).
    exact (IH _ _ Htail).
Qed.

(* A derivation of the relation from a value that holds no array value
   is one by those rules: the two rules of the array have an array
   value for their first value, and the others relate the parts of the
   first value. *)
Fixpoint value_lifted_by_no_array_rule (statics' : static_env)
    (v v' : value) (H : value_lifted statics' v v') {struct H} :
  no_array_value v = true -> value_lifted_no_array statics' v v'.
Proof.
  destruct H
    as [ c | s | c | tag fields fields' Hfields | items items' Hitems
       | code slots slots' Hslots | fslot decls slots slots' Hslots
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | c s Hlookup | tag fields fields' s Hlookup Hfields
       | items items' s Hlookup Hitems
       | fslot decls slots slots' s syms Hsym Hsyms Hbound Hslots
       | code slots slots' s Hlookup Hslots | l ];
    simpl; intros Hno; try discriminate Hno.
  - apply No_array_const.
  - apply No_array_symbol.
  - apply No_array_boxed_number.
  - apply No_array_block.
    revert fields fields' Hfields Hno. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + exact (value_lifted_by_no_array_rule _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply No_array_closure.
    revert slots slots' Hslots Hno. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + split; [ exact Hkey | ].
      exact (value_lifted_by_no_array_rule _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply No_array_closure_of_set.
    revert slots slots' Hslots Hno. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + split; [ exact Hkey | ].
      exact (value_lifted_by_no_array_rule _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply andb_true_iff in Hno as [Hcallee Hargs].
    apply No_array_partial_application;
      [ exact (value_lifted_by_no_array_rule _ _ _ Hpcallee Hcallee) | ].
    revert pargs pargs' Hpargs Hargs. fix IH 3.
    intros pargs pargs' [ | w w' rest rest' Hhead Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + exact (value_lifted_by_no_array_rule _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply No_array_to_boxed_number. exact Hlookup.
  - apply (No_array_to_block _ tag fields fields' s Hlookup).
    clear Hlookup. revert fields fields' Hfields Hno. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + exact (value_lifted_by_no_array_rule _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply (No_array_to_closure _ fslot decls slots slots' s syms Hsym Hsyms
             Hbound).
    clear Hbound. revert slots slots' Hslots Hno. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + split; [ exact Hkey | ].
      exact (value_lifted_by_no_array_rule _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply (No_array_to_unnamed_closure _ code slots slots' s Hlookup).
    clear Hlookup. revert slots slots' Hslots Hno. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; simpl;
      intros Hall; constructor;
      apply andb_true_iff in Hall as [Hw Hrest].
    + split; [ exact Hkey | ].
      exact (value_lifted_by_no_array_rule _ _ _ Hhead Hw).
    + exact (IH _ _ Htail Hrest).
  - apply No_array_location.
Qed.

Theorem value_lifted_no_array_exact : value_lifted_no_array_exact_statement.
Proof.
  intros statics' v v'. split.
  - intros H. split.
    + exact (value_lifted_no_array_lifted _ _ _ H).
    + exact (value_lifted_no_array_holds_none _ _ _ H).
  - intros [H Hno]. exact (value_lifted_by_no_array_rule _ _ _ H Hno).
Qed.

Theorem lift_simulates_no_array : lift_simulates_no_array_statement.
Proof.
  intros u lets known e. cbv zeta. intros Hwf Honly Hmakes k vs hp Hexit.
  assert (Hfree :
    (∀ v : value, v ∈ vs -> no_array_value v = true)
    ∧ heap_no_array_value hp).
  { destruct Hexit as [fuel [rd Hrun]].
    pose proof
      (run_free_of_arrays fuel _ (unit_state_free_of_arrays e Hmakes))
      as Hfree.
    rewrite Hrun in Hfree. exact Hfree. }
  destruct Hfree as [Hvs Hhp].
  pose proof (lift_simulates u lets known e) as Hsimulates.
  cbv zeta in Hsimulates.
  destruct (proj1 (Hsimulates Hwf Honly) k vs hp Hexit)
    as (n & last & vs' & hp' & rd & Hrun & Hstep & Hlifted & Hheap).
  exists n, last, vs', hp', rd.
  split; [ exact Hrun | split; [ exact Hstep | ] ].
  split.
  - clear Hrun Hstep Hexit Hheap.
    induction Hlifted as [ | v v' vs vs' Hv _ IH]; constructor.
    + exact (value_lifted_by_no_array_rule _ _ _ Hv
               (Hvs v (or_introl eq_refl))).
    + apply IH. intros w Hw. exact (Hvs w (or_intror Hw)).
  - exact (heap_alike_no_array _ _ hp hp'
             (value_lifted_by_no_array_rule _) Hhp Hheap).
Qed.
