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

(* Proofs of the statements of FlambdaHeapMachineSpec.v. No review
   needed. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaValueProofs FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaUnusedStaticsSpec FlambdaClosuresSpec FlambdaHeapSpec
  FlambdaHeapMachineSpec FlambdaHeapProofs.
From OxCamlFlambda2 Require FlambdaNameAbstractionSpec.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs.
From OxCamlFlambda2 Require FlambdaSemanticsProofs.
Import ListNotations.

(* FlambdaPrimitiveProofs.v makes [eval_prim] opaque for the files that
   require it, and FlambdaSemanticsProofs.v [step]. The proofs here run
   the machine on terms, so they need both to compute. *)
#[local] Transparent eval_prim.
#[local] Transparent step.

Local Open Scope Z_scope.

Theorem fresh_location_fresh_name : fresh_location_fresh_name_statement.
Proof. intros h. reflexivity. Qed.

(* An array that holds a partial application that holds a location: the
   primitives of arrays, the calls and the heap together. [get cell _]
   is the first field of [cell], as it is when the second argument
   comes. *)
Module Array_of_a_partial_application.

Definition x : variable := 1%positive.
Definition y : variable := 2%positive.
Definition my_closure : variable := 3%positive.
Definition partial : variable := 4%positive.
Definition r : variable := 5%positive.
Definition a : variable := 6%positive.
Definition u : variable := 7%positive.
Definition loaded : variable := 8%positive.
Definition z : variable := 9%positive.
Definition ret : continuation := 1%positive.
Definition exn : continuation := 2%positive.
Definition done : continuation := 1%positive.
Definition error : continuation := 2%positive.
Definition after_partial : continuation := 3%positive.
Definition get_code_id : code_id := 1%positive.
Definition get_symbol : symbol := 1%positive.

Definition one_value : arity := [Singleton Value].
Definition imm (i : Z) : value := Value_const (Tagged_immediate i).
Definition any_field : block_access_kind :=
  Block_access_values None None Any_value.

Definition get_code : code :=
  Code [x; y] my_closure ret exn
    (make_code_metadata [Singleton Value; Singleton Value] one_value)
    (Let z Value (Prim (Unary (Block_load any_field Mutable 0)) [Var x])
       (Apply_cont (Apply_cont_expr ret [Var z] None))).

(* What is done with a partial application of [get] to the cell [r]: it
   is stored in an array, 8 is written into the cell, and the element
   of the array is loaded and given its second argument. *)
Definition use_partial : expr :=
  Let a Value
    (Prim (Heap (Make_mutable_array Array_values))
       [Const (Tagged_immediate 0)])
    (Let u Value
       (Prim (Heap (Array_set Array_values (Set_values Assignment)))
          [Var a; Const (Tagged_immediate 0); Var partial])
       (Let u Value
          (Prim (Heap (Block_set any_field Assignment 0))
             [Var r; Const (Tagged_immediate 8)])
          (Let loaded Value
             (Prim (Binary (Array_load Array_values Load_values Mutable))
                [Var a; Const (Tagged_immediate 0)])
             (Apply
                (Apply_expr (Var loaded) Indirect
                   [Const (Tagged_immediate 0)] done error [] one_value
                   one_value))))).

(* The cell is made holding 7, and [get] is given it alone. *)
Definition program : expr :=
  Let_static
    [Bind_code get_code_id get_code;
     Bind_symbol get_symbol (Static_const_set_of_closures get_code_id [])]
    (Let r Value
       (Prim (Heap (Make_block_with_identity Cell_mutable 0))
          [Const (Tagged_immediate 7)])
       (Let_cont_non_recursive after_partial
          (Handler [partial] (make_handler_metadata [Value]) use_partial)
          (Apply
             (Apply_expr (Symbol get_symbol) Indirect [Var r] after_partial
                error [] one_value one_value)))).

(* It returns 8: the array holds the partial application with the
   location of the cell, and the call completed reads the cell as it is
   then. *)
Example returns_what_the_cell_holds_at_the_end :
  ∃ o : outcome,
    heap_run 11 (Heap_state program [] [] empty_static_env empty_heap)
    = Heap_exit done o
    ∧ outcome_values o = [imm 8].
Proof. eexists. split; reflexivity. Qed.

(* The same use from a state whose variable [partial] is bound to a
   partial application of [get] to the location [l]. *)
Definition a_cell : location := 1%positive.
Definition dangling : location := 9%positive.
Definition h : heap := [(a_cell, Cell_block Cell_mutable 0 [imm 7])].
Definition statics : static_env := Static_env [(get_code_id, get_code)] [].

Definition partial_of (l : location) : value :=
  Value_partial_application (Value_closure get_code_id []) [Value_location l]
    one_value.

Definition use (l : location) : heap_state :=
  Heap_state use_partial
    [(partial, partial_of l); (r, Value_location a_cell)] [] statics h.

Example use_of_a_cell :
  ∃ o : outcome,
    heap_run 7 (use a_cell) = Heap_exit done o ∧ outcome_values o = [imm 8].
Proof. eexists. split; reflexivity. Qed.

(* A partial application that holds a location bound to nothing is not
   refused where it is stored: [exec_prim] refuses an argument that is
   itself such a location ([argument_bound]), and does not look inside
   the others. *)
Example a_dangling_location_is_not_stored :
  exec_prim statics h (Heap (Make_mutable_array Array_values))
    [Value_location dangling]
  = None.
Proof. reflexivity. Qed.

Example a_partial_application_of_one_is_stored :
  is_some
    (exec_prim statics h (Heap (Make_mutable_array Array_values))
       [partial_of dangling])
  = true.
Proof. reflexivity. Qed.

(* It is refused where it is used: the fifth step is the call that
   completes the partial application, and the machine is undefined at
   the load in the body of [get], whose argument is the location. *)
Example completed_it_is_undefined :
  ∃ s : heap_state,
    heap_run 5 (use dangling) = Heap_step s
    ∧ hs_expr s = code_body get_code
    ∧ heap_step s = Heap_undefined.
Proof. eexists. split; [ reflexivity | ]. split; reflexivity. Qed.

(* And no state reached from a closed one holds such a value
   ([heap_step_closed_statement]): this one is not closed. *)
Example it_is_not_closed : ~ state_closed (use dangling).
Proof.
  intros (_ & Hvars & _).
  specialize
    (Hvars partial (partial_of dangling) (or_introl eq_refl) dangling
       (or_introl eq_refl)).
  discriminate Hvars.
Qed.

End Array_of_a_partial_application.

(* ======================================================================== *)
(* The machine                                                              *)
(* ======================================================================== *)

Ltac break_match :=
  match goal with
  | |- context [match ?x with _ => _ end] => destruct x eqn:?
  end.

(* The same, on a scrutinee that has no match in it. *)
Ltac break_innermost :=
  match goal with
  | |- context [match ?x with _ => _ end] =>
    lazymatch x with
    | context [match _ with _ => _ end] => fail
    | _ => destruct x eqn:?
    end
  end.

Lemma heap_exec_named_evolves (statics : static_env) (h h' : heap) (vars : env)
    (d : named) (v : value) :
  heap_exec_named statics h vars d = Some (v, h') -> heap_evolves h h'.
Proof.
  assert (Hpure :
            (let* v := eval_named statics vars d in Some (v, h)) = Some (v, h')
            -> heap_evolves h h').
  { destruct (eval_named statics vars d); simpl; [ | discriminate ].
    intros [= _ <-]. apply heap_evolves_refl. }
  destruct d as [s | p args | | | | | ]; try exact Hpure.
  simpl. destruct (eval_simples vars args) as [vs | ]; simpl;
    [ | discriminate ].
  apply exec_prim_evolves.
Qed.

(* A call leaves the heap as it is. *)
Lemma heap_call_code_heap (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (h : heap) (s' : heap_state) :
  heap_call_code closure c vs args_arity return_arity return_to exn_to statics
    h
  = Heap_step s' ->
  hs_heap s' = h.
Proof.
  unfold heap_call_code. repeat break_match; try discriminate.
  intros [= <-]. reflexivity.
Qed.

Lemma heap_run_call_heap (return_partial : value -> heap_step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (h : heap)
    (s' : heap_state) :
  (∀ (partial : value) (s' : heap_state),
    return_partial partial = Heap_step s' -> hs_heap s' = h) ->
  heap_run_call return_partial shape return_arity return_to exn_to statics h
  = Heap_step s' ->
  hs_heap s' = h.
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply heap_call_code_heap.
  - apply heap_call_code_heap.
  - apply Hpartial.
  - discriminate.
Qed.

(* Applying a continuation leaves the heap as it is. *)
Lemma heap_apply_cont_value_heap (c : cont_value) (vs : list value)
    (statics : static_env) (h : heap) (s' : heap_state) :
  heap_apply_cont_value c vs statics h = Heap_step s' -> hs_heap s' = h.
Proof.
  revert vs s'.
  induction c
    as [handler vars conts recursive | k | c IH extra
       | remaining remaining_arity return_arity return_to IHreturn exn_to
           IHexn];
    intros vs s'; simpl.
  - destruct handler as [params metadata handler].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs vars); [ | discriminate ].
    intros [= <-]. reflexivity.
  - discriminate.
  - apply IH.
  - destruct vs as [ | v [ | v2 vs]]; try discriminate.
    apply heap_run_call_heap. intros partial s0. apply IHreturn.
Qed.

Lemma heap_call_function_heap (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (h : heap)
    (s' : heap_state) :
  heap_call_function callee kind vs args_arity return_arity return_to exn_to
    statics h
  = Heap_step s' ->
  hs_heap s' = h.
Proof.
  unfold heap_call_function. apply heap_run_call_heap. intros partial s0.
  apply heap_apply_cont_value_heap.
Qed.

(* What the machine with a heap does and what the machine without does,
   where they agree. *)
Definition results_agree (statics : static_env) (h : heap)
    (result : heap_step_result) (result' : step_result) : Prop :=
  match result, result' with
  | Heap_step s', Step t => forget_heap s' = t ∧ hs_heap s' = h
  | Heap_exit k o, Exit k' vs' _ => k = k' ∧ o = Outcome vs' statics h
  | Heap_undefined, Undefined => True
  | _, _ => False
  end.

(* A call is what FlambdaSemanticsSpec.call_code and run_call are. *)
Lemma heap_call_code_agrees (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (h : heap) :
  results_agree statics h
    (heap_call_code closure c vs args_arity return_arity return_to exn_to
       statics h)
    (call_code closure c vs args_arity return_arity return_to exn_to statics h).
Proof.
  unfold results_agree, heap_call_code, call_code.
  repeat (break_innermost; simpl; auto).
Qed.

Lemma heap_run_call_agrees (return_partial : value -> heap_step_result)
    (return_partial' : value -> step_result) (shape : option call_shape)
    (return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (h : heap) :
  (∀ partial : value,
    results_agree statics h (return_partial partial)
      (return_partial' partial)) ->
  results_agree statics h
    (heap_run_call return_partial shape return_arity return_to exn_to statics
       h)
    (run_call return_partial' shape return_arity return_to exn_to statics h).
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply heap_call_code_agrees.
  - apply heap_call_code_agrees.
  - apply Hpartial.
  - exact I.
Qed.

(* Applying a continuation is what FlambdaSemanticsSpec.apply_cont_value
   is. *)
Lemma heap_apply_cont_value_agrees (c : cont_value) (vs : list value)
    (statics : static_env) (h : heap) :
  results_agree statics h (heap_apply_cont_value c vs statics h)
    (apply_cont_value c vs statics h).
Proof.
  revert vs.
  induction c
    as [handler vars conts recursive | k | c IH extra
       | remaining remaining_arity return_arity return_to IHreturn exn_to
           IHexn];
    intros vs; simpl.
  - destruct handler as [params metadata handler].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars); simpl; auto.
  - auto.
  - apply IH.
  - destruct vs as [ | v [ | v2 vs]]; try exact I.
    apply heap_run_call_agrees. intros partial. apply IHreturn.
Qed.

Lemma heap_call_function_agrees (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (h : heap) :
  results_agree statics h
    (heap_call_function callee kind vs args_arity return_arity return_to
       exn_to statics h)
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics h).
Proof.
  unfold heap_call_function, call_function. apply heap_run_call_agrees.
  intros partial. apply heap_apply_cont_value_agrees.
Qed.

Theorem heap_step_evolves : heap_step_evolves_statement.
Proof.
  intros [e vars conts statics h] s'.
  destruct e
    as [x x_kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ]; simpl.
  - destruct (heap_exec_named statics h vars d) as [[v h'] | ] eqn:Hexec;
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) x_kind); [ | discriminate ].
    intros [= <-]. simpl. exact (heap_exec_named_evolves _ _ _ _ _ _ Hexec).
  - repeat break_match; try discriminate. intros [= <-].
    apply heap_evolves_refl.
  - intros [= <-]. apply heap_evolves_refl.
  - repeat break_match; try discriminate. intros [= <-].
    apply heap_evolves_refl.
  - repeat break_match; try discriminate.
    intros Hstep.
    rewrite (heap_call_function_heap _ _ _ _ _ _ _ _ _ _ Hstep).
    apply heap_evolves_refl.
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    intros Hstep. rewrite (heap_apply_cont_value_heap _ _ _ _ _ Hstep).
    apply heap_evolves_refl.
  - repeat break_match; try discriminate. intros [= <-].
    apply heap_evolves_refl.
  - discriminate.
Qed.

Theorem heap_step_agrees : heap_step_agrees_statement.
Proof.
  intros [e vars conts statics h].
  unfold forget_heap. simpl.
  destruct e
    as [x x_kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ]; simpl.
  - destruct d; try discriminate; intros _; unfold exec_named; simpl;
      repeat (break_innermost; simpl; auto).
  - intros _. destruct (eval_static_bindings vars bindings statics); auto.
  - intros _. auto.
  - intros _. destruct (handlers_bind_statics handlers); auto.
  - intros _.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; simpl; auto.
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; simpl; auto.
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]; simpl;
      auto.
    apply heap_call_function_agrees.
  - intros _. unfold apply_continuation.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    apply heap_apply_cont_value_agrees.
  - intros _. repeat (break_innermost; simpl; auto).
  - intros _. exact I.
Qed.

(* ======================================================================== *)
(* The relation                                                             *)
(* ======================================================================== *)

Lemma Forall2_same {A : Type} (R : A -> A -> Prop) (l : list A) :
  (∀ x : A, x ∈ l -> R x x) -> Forall2 R l l.
Proof.
  induction l as [ | x l IH]; intros Hall; constructor.
  - apply Hall. left. reflexivity.
  - apply IH. intros y Hy. apply Hall. right. exact Hy.
Qed.

Lemma Forall2_map_same {A B : Type} (R : B -> B -> Prop) (f : A -> B)
    (l : list A) :
  (∀ x : A, x ∈ l -> R (f x) (f x)) -> Forall2 R (map f l) (map f l).
Proof.
  intros Hall. apply Forall2_same. intros y Hy.
  apply in_map_iff in Hy as (x & <- & Hx). exact (Hall x Hx).
Qed.

Lemma Forall2_mono {A B : Type} (R R' : A -> B -> Prop) (l : list A)
    (l' : list B) :
  (∀ (x : A) (y : B), R x y -> R' x y) -> Forall2 R l l' -> Forall2 R' l l'.
Proof.
  intros Hmono Hall. induction Hall; constructor; auto.
Qed.

Lemma Forall2_compose {A B C : Type} (R : A -> B -> Prop) (R' : B -> C -> Prop)
    (l : list A) (l' : list B) (l'' : list C) :
  Forall2 R l l' ->
  Forall2 R' l' l'' ->
  Forall2 (fun x z => ∃ y : B, R x y ∧ R' y z) l l''.
Proof.
  intros Hall. revert l''.
  induction Hall as [ | x y l l' Hxy Hall IH]; intros l'' Hall';
    inversion Hall' as [ | y' z l0 l0'' Hyz Hall'']; subst; constructor.
  - exists y. auto.
  - apply IH. exact Hall''.
Qed.

(* One step of the relation is monotone in what relates the contents. *)
Lemma value_step_mono (poison_refines : bool) (r : correspondence)
    (statics statics' : static_env) (R R' : value -> value -> Prop)
    (v v' : value) :
  (∀ a a' : value, R a a' -> R' a a') ->
  value_step poison_refines r statics statics' R v v' ->
  value_step poison_refines r statics statics' R' v v'.
Proof.
  intros Hmono Hstep.
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements].
  - apply Step_const.
  - apply Step_location. exact Hl.
  - exact
      (Step_block _ _ _ _ _ _ _ tag fields fields' Hv Hv'
         (Forall2_mono _ _ _ _ Hmono Hfields)).
  - exact (Step_boxed_number _ _ _ _ _ _ _ c c' Hv Hv' Hc).
  - apply Step_same_boxed_number.
  - exact (Step_string _ _ _ _ _ _ _ bytes Hv Hv').
  - apply
      (Step_closure _ _ _ _ _ _ _ fslot fdecls slots slots' Hv Hv' Hnames
         (Forall2_mono _ _ _ _ Hmono Hslots)).
    destruct Hsiblings as [Hbound | [Hsiblings Hnone]];
      [ left; exact Hbound | ].
    right. split; [ | exact Hnone ]. intros move_to w Hw.
    destruct (Hsiblings _ _ Hw) as (w' & Hw' & Hrelated).
    exists w'. split; [ exact Hw' | exact (Hmono _ _ Hrelated) ].
  - apply Step_same_closure_of_set. exact Hgslot.
  - apply Step_opaque_symbol; assumption.
  - apply Step_poison; assumption.
  - apply Step_partial_application;
      [ exact (Hmono _ _ Hpcallee)
      | exact (Forall2_mono _ _ _ _ Hmono Hpargs) ].
  - exact
      (Step_array _ _ _ _ _ _ _ elements elements' Hv Hv'
         (Forall2_mono _ _ _ _ Hmono Helements)).
Qed.

(* Induction on the relation, through the lists of related contents. *)
Lemma value_related_induction (poison_refines : bool) (r : correspondence)
    (statics statics' : static_env) (P : value -> value -> Prop) :
  (∀ v v' : value, corr_values r v v' -> P v v') ->
  (∀ v v' : value,
    value_step poison_refines r statics statics'
      (fun a a' =>
         value_related poison_refines r statics statics' a a' ∧ P a a')
      v v' ->
    P v v') ->
  ∀ v v' : value,
    value_related poison_refines r statics statics' v v' -> P v v'.
Proof.
  intros Hcorr Hstep. fix IH 3. intros v v' Hrelated.
  destruct Hrelated as [v v' Hv | v v' Hv]; [ exact (Hcorr v v' Hv) | ].
  apply Hstep.
  destruct Hv
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements].
  - apply Step_const.
  - apply Step_location. exact Hl.
  - apply (Step_block _ _ _ _ _ _ _ tag fields fields' Hv Hv').
    exact
      ((fix go (vs vs' : list value)
          (H : Forall2 (value_related poison_refines r statics statics')
                 vs vs')
          : Forall2
              (fun a a' =>
                 value_related poison_refines r statics statics' a a'
                 ∧ P a a')
              vs vs' :=
          match H with
          | Forall2_nil _ => Forall2_nil _
          | Forall2_cons _ _ Hhead Htail =>
            Forall2_cons _ _ (conj Hhead (IH _ _ Hhead)) (go _ _ Htail)
          end)
         _ _ Hfields).
  - exact (Step_boxed_number _ _ _ _ _ _ _ c c' Hv Hv' Hc).
  - apply Step_same_boxed_number.
  - exact (Step_string _ _ _ _ _ _ _ bytes Hv Hv').
  - apply (Step_closure _ _ _ _ _ _ _ fslot fdecls slots slots' Hv Hv'
      Hnames).
    + exact
        ((fix go (vs vs' : list value)
            (H : Forall2 (value_related poison_refines r statics statics')
                   vs vs')
            : Forall2
                (fun a a' =>
                   value_related poison_refines r statics statics' a a'
                   ∧ P a a')
                vs vs' :=
            match H with
            | Forall2_nil _ => Forall2_nil _
            | Forall2_cons _ _ Hhead Htail =>
              Forall2_cons _ _ (conj Hhead (IH _ _ Hhead)) (go _ _ Htail)
            end)
           _ _ Hslots).
    + destruct Hsiblings as [Hbound | [Hsiblings Hnone]];
        [ left; exact Hbound | ].
      right. split; [ | exact Hnone ]. intros move_to w Hw.
      destruct (Hsiblings move_to w Hw) as (w' & Hw' & Hrelated).
      exists w'. split; [ exact Hw' | ].
      exact (conj Hrelated (IH _ _ Hrelated)).
  - apply Step_same_closure_of_set. exact Hgslot.
  - apply Step_opaque_symbol; assumption.
  - apply Step_poison; assumption.
  - apply Step_partial_application;
      [ exact (conj Hpcallee (IH _ _ Hpcallee)) | ].
    exact
      ((fix go (vs vs' : list value)
          (H : Forall2 (value_related poison_refines r statics statics')
                 vs vs')
          : Forall2
              (fun a a' =>
                 value_related poison_refines r statics statics' a a'
                 ∧ P a a')
              vs vs' :=
          match H with
          | Forall2_nil _ => Forall2_nil _
          | Forall2_cons _ _ Hhead Htail =>
            Forall2_cons _ _ (conj Hhead (IH _ _ Hhead)) (go _ _ Htail)
          end)
         _ _ Hpargs).
  - apply (Step_array _ _ _ _ _ _ _ elements elements' Hv Hv').
    exact
      ((fix go (vs vs' : list value)
          (H : Forall2 (value_related poison_refines r statics statics')
                 vs vs')
          : Forall2
              (fun a a' =>
                 value_related poison_refines r statics statics' a a'
                 ∧ P a a')
              vs vs' :=
          match H with
          | Forall2_nil _ => Forall2_nil _
          | Forall2_cons _ _ Hhead Htail =>
            Forall2_cons _ _ (conj Hhead (IH _ _ Hhead)) (go _ _ Htail)
          end)
         _ _ Helements).
Qed.

(* Two related values have their heads in common, under a valid
   correspondence. *)
Lemma value_related_step (poison_refines : bool) (r : correspondence)
    (statics statics' : static_env) (h h' : heap) (v v' : value) :
  correspondence_valid poison_refines r statics statics' h h' ->
  value_related poison_refines r statics statics' v v' ->
  value_step poison_refines r statics statics'
    (value_related poison_refines r statics statics') v v'.
Proof.
  intros (_ & _ & Hvalues) [v0 v0' Hcorr | v0 v0' Hstep];
    [ exact (Hvalues _ _ Hcorr) | exact Hstep ].
Qed.

(* The rules of the relation, by name. *)
Section Rules.

Variable poison_refines : bool.
Variable r : correspondence.
Variables statics statics' : static_env.

#[local] Abbreviation related :=
  (value_related poison_refines r statics statics').

Lemma related_const (c : const) : related (Value_const c) (Value_const c).
Proof. apply Related_step, Step_const. Qed.

Lemma related_location (l l' : location) :
  corr_locations r l l' -> related (Value_location l) (Value_location l').
Proof. intros Hl. apply Related_step, Step_location. exact Hl. Qed.

Lemma related_block (v v' : value) (tag : Z) (fields fields' : list value) :
  block_of statics v = Some (tag, fields) ->
  block_of statics' v' = Some (tag, fields') ->
  Forall2 related fields fields' ->
  related v v'.
Proof.
  intros Hv Hv' Hfields. apply Related_step.
  exact (Step_block _ _ _ _ _ _ _ tag fields fields' Hv Hv' Hfields).
Qed.

Lemma related_closure (v v' : value) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (slots slots' : list (value_slot * value)) :
  closure_in_set statics v = Some (slot, decls, slots) ->
  closure_in_set statics' v' = Some (slot, decls, slots') ->
  map fst slots = map fst slots' ->
  Forall2 related (map snd slots) (map snd slots') ->
  (set_bound statics v ∧ set_bound statics' v')
  ∨ ((∀ (move_to : function_slot) (w : value),
       project_function_slot statics slot move_to v = Some w ->
       ∃ w' : value,
         project_function_slot statics' slot move_to v' = Some w'
         ∧ related w w')
     ∧ (∀ move_to : function_slot,
         project_function_slot statics slot move_to v = None ->
         project_function_slot statics' slot move_to v' = None)) ->
  related v v'.
Proof.
  intros Hv Hv' Hnames Hslots Hsiblings. apply Related_step.
  exact
    (Step_closure _ _ _ _ _ _ _ slot decls slots slots' Hv Hv' Hnames Hslots
       Hsiblings).
Qed.

End Rules.

(* What an observation says of a value excludes the others. *)
Lemma not_allocated (statics : static_env) (v : value) :
  is_allocated statics v = false ->
  closure_of statics v = None
  ∧ block_of statics v = None
  ∧ boxed_number_of statics v = None
  ∧ string_of statics v = None.
Proof.
  unfold is_allocated. rewrite !orb_false_iff.
  intros [[[[[Hclosure Hblock] Hboxed] Hstring] _] _].
  destruct (closure_of statics v); [ discriminate | ].
  destruct (block_of statics v); [ discriminate | ].
  destruct (boxed_number_of statics v); [ discriminate | ].
  destruct (string_of statics v); [ discriminate | ].
  auto.
Qed.

Lemma not_allocated_array (statics : static_env) (v : value) :
  is_allocated statics v = false -> array_of statics v = None.
Proof.
  unfold is_allocated. rewrite !orb_false_iff.
  intros [[_ Harray] _].
  destruct (array_of statics v); [ discriminate | reflexivity ].
Qed.

Lemma boxable_contents_same (c c' : const) :
  boxable_contents c = Some c' -> c' = c.
Proof.
  unfold boxable_contents. destruct (boxable_number_of_const c);
    [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

(* A closure of a set is a closure, at the code of its slot; and the
   other way round. *)
Lemma closure_in_set_closure_of (statics : static_env) (v : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  closure_in_set statics v = Some (slot, decls, slots) ->
  ∃ code : code_id,
    lookup slot decls = Some code ∧ closure_of statics v = Some (code, slots).
Proof.
  destruct v as [c | code slots' | slot' decls' slots' | s | tag fields
    | elements | c | l | callee0 applied applied_arity];
    simpl; try discriminate.
  - intros [= <- <- <-]. exists code. rewrite lookup_head. auto.
  - destruct (lookup slot' decls') as [code | ] eqn:Hslot; simpl;
      [ | discriminate ].
    intros [= <- <- <-]. exists code. auto.
  - destruct (lookup s (static_values statics))
      as [[code slots' | slot' syms decls' slots' | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + intros [= <- <- <-]. exists code. rewrite lookup_head. auto.
    + destruct (lookup slot' decls') as [code | ] eqn:Hslot; simpl;
        [ | discriminate ].
      intros [= <- <- <-]. exists code. auto.
Qed.

Lemma closure_of_closure_in_set (statics : static_env) (v : value)
    (code : code_id) (slots : list (value_slot * value)) :
  closure_of statics v = Some (code, slots) ->
  ∃ (slot : function_slot) (decls : list (function_slot * code_id)),
    closure_in_set statics v = Some (slot, decls, slots)
    ∧ lookup slot decls = Some code.
Proof.
  destruct v as [c | code' slots' | slot decls slots' | s | tag fields
    | elements | c | l | callee0 applied applied_arity];
    simpl; try discriminate.
  - intros [= <- <-]. eexists _, _. split; [ reflexivity | ].
    apply lookup_head.
  - destruct (lookup slot decls) as [code' | ] eqn:Hslot; simpl;
      [ | discriminate ].
    intros [= <- <-]. exists slot, decls. auto.
  - destruct (lookup s (static_values statics))
      as [[code' slots' | slot syms decls slots' | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + intros [= <- <-]. eexists _, _. split; [ reflexivity | ].
      apply lookup_head.
    + destruct (lookup slot decls) as [code' | ] eqn:Hslot; simpl;
        [ | discriminate ].
      intros [= <- <-]. exists slot, decls. auto.
Qed.

(* Two lists of bindings with the same names bind the same names. *)
Lemma lookup_same_keys {A B : Type} (k : positive) (a : list (positive * A))
    (b : list (positive * B)) :
  map fst a = map fst b -> lookup k a = None <-> lookup k b = None.
Proof.
  intros Hkeys. pose proof (is_bound_among k a) as Ha.
  pose proof (is_bound_among k b) as Hb. rewrite Hkeys, <- Hb in Ha.
  unfold is_bound in Ha.
  destruct (lookup k a), (lookup k b); try discriminate; split; intros H;
    try discriminate H; reflexivity.
Qed.

(* A closure whose set is known projects, to each function slot of its
   set, a closure of the same set at that slot, whose set is known; and
   to no other slot. *)
Lemma project_set_bound (st : static_env) (v : value) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) (move_to : function_slot) :
  set_bound st v ->
  closure_in_set st v = Some (slot, decls, slots) ->
  (project_function_slot st slot move_to v = None
   <-> lookup move_to decls = None)
  ∧ (∀ x : value,
      project_function_slot st slot move_to v = Some x ->
      set_bound st x ∧ closure_in_set st x = Some (move_to, decls, slots)).
Proof.
  destruct v as [c | code vs | slot' decls' vs | s | tag fields
    | elements | c | l | callee0 applied applied_arity];
    cbn -[Pos.eqb]; try discriminate.
  - intros _ [= <- <- <-].
    destruct (Pos.eqb_spec move_to unnamed_function_slot) as [-> | Hne].
    + rewrite lookup_head. split; [ split; discriminate | ].
      intros x [= <-]. split; [ exact I | reflexivity ].
    + rewrite (lookup_other _ _ _ _ Hne).
      split; [ split; reflexivity | discriminate ].
  - intros _.
    destruct (lookup slot' decls') as [code | ] eqn:Hslot; cbn -[Pos.eqb];
      [ | discriminate ].
    intros [= <- <- <-]. rewrite Pos.eqb_refl.
    destruct (lookup move_to decls') as [code' | ] eqn:Hto; cbn -[Pos.eqb].
    + split; [ split; discriminate | ].
      intros x [= <-]. split; [ exact I | ].
      cbn -[Pos.eqb]. rewrite Hto. reflexivity.
    + split; [ split; reflexivity | discriminate ].
  - destruct (lookup s (static_values st))
      as [[code vs | slot' syms decls' vs | tag fields | contents | bytes
          | elements | ] | ]
      eqn:Hlookup; try discriminate.
    + intros _ [= <- <- <-].
      destruct (Pos.eqb_spec move_to unnamed_function_slot) as [-> | Hne].
      * rewrite lookup_head. split; [ split; discriminate | ].
        intros x [= <-]. cbn -[Pos.eqb]. rewrite Hlookup.
        split; [ exact I | reflexivity ].
      * rewrite (lookup_other _ _ _ _ Hne).
        split; [ split; reflexivity | discriminate ].
    + intros [Hkeys Hbound].
      destruct (lookup slot' decls') as [code | ] eqn:Hslot; cbn -[Pos.eqb];
        [ | discriminate ].
      intros [= <- <- <-]. rewrite Pos.eqb_refl.
      destruct (lookup move_to decls') as [code' | ] eqn:Hto; cbn -[Pos.eqb].
      * destruct (lookup move_to syms) as [t | ] eqn:Ht; cbn -[Pos.eqb].
        -- split; [ split; discriminate | ].
           intros x [= <-].
           pose proof (Hbound move_to t (lookup_in _ _ _ Ht)) as Ht_bound.
           cbn -[Pos.eqb]. rewrite Ht_bound. cbn -[Pos.eqb]. rewrite Hto.
           split; [ split; assumption | reflexivity ].
        -- exfalso.
           apply (lookup_same_keys move_to syms decls' Hkeys) in Ht.
           rewrite Ht in Hto. discriminate.
      * split; [ split; reflexivity | discriminate ].
Qed.

(* So two closures whose sets are known, at one slot of sets with the
   same functions, project to such closures at the other slots. *)
Lemma project_set_bound_both (st st' : static_env) (v v' : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (slots slots' : list (value_slot * value)) (move_to : function_slot)
    (x : value) :
  set_bound st v ->
  set_bound st' v' ->
  closure_in_set st v = Some (slot, decls, slots) ->
  closure_in_set st' v' = Some (slot, decls, slots') ->
  project_function_slot st slot move_to v = Some x ->
  ∃ x' : value,
    project_function_slot st' slot move_to v' = Some x'
    ∧ (set_bound st x ∧ set_bound st' x')
    ∧ closure_in_set st x = Some (move_to, decls, slots)
    ∧ closure_in_set st' x' = Some (move_to, decls, slots').
Proof.
  intros Hbound Hbound' Hclosure Hclosure' Hx.
  destruct (project_set_bound _ _ _ _ _ move_to Hbound Hclosure)
    as [Hnone Hsome].
  destruct (project_function_slot st' slot move_to v') as [x' | ] eqn:Hx';
    destruct (project_set_bound _ _ _ _ _ move_to Hbound' Hclosure')
      as [Hnone' Hsome'].
  - exists x'. destruct (Hsome x Hx) as [Hx_bound Hx_closure].
    destruct (Hsome' x' Hx') as [Hx'_bound Hx'_closure]. auto.
  - apply Hnone' in Hx'. apply Hnone in Hx'. rewrite Hx' in Hx.
    discriminate.
Qed.

(* A projection of a function slot from a symbol is a symbol. *)
Lemma project_symbol (st : static_env) (s : symbol) (w : value)
    (move_from move_to : function_slot) :
  project_function_slot st move_from move_to (Value_symbol s) = Some w ->
  ∃ t : symbol, w = Value_symbol t.
Proof.
  cbn -[Pos.eqb].
  destruct (lookup s (static_values st))
    as [[code vs | slot syms decls vs | tag fields | contents | bytes
         | elements | ] | ];
    try discriminate.
  - destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. eauto.
  - destruct (Pos.eqb slot move_from); [ | discriminate ].
    destruct (lookup slot decls); simpl; [ | discriminate ].
    destruct (lookup move_to decls); simpl; [ | discriminate ].
    destruct (lookup move_to syms); simpl; [ | discriminate ].
    intros [= <-]. eauto.
Qed.

(* A projection of a function slot is from a closure at that slot. *)
Lemma project_function_slot_closure (st : static_env) (v w : value)
    (move_from move_to : function_slot) :
  project_function_slot st move_from move_to v = Some w ->
  ∃ (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)),
    closure_in_set st v = Some (move_from, decls, slots).
Proof.
  destruct v as [c | code vs | slot decls vs | s | tag fields | elements | c | l
    | callee0 applied applied_arity];
    cbn -[Pos.eqb]; try discriminate.
  - destruct (Pos.eqb_spec move_from unnamed_function_slot) as [-> | _];
      [ | discriminate ].
    intros _. eauto.
  - destruct (Pos.eqb_spec slot move_from) as [-> | _]; [ | discriminate ].
    destruct (lookup move_from decls); simpl; [ | discriminate ].
    intros _. eauto.
  - destruct (lookup s (static_values st))
      as [[code vs | slot syms decls vs | tag fields | contents | bytes
           | elements | ] | ];
      try discriminate.
    + destruct (Pos.eqb_spec move_from unnamed_function_slot) as [-> | _];
        [ | discriminate ].
      intros _. eauto.
    + destruct (Pos.eqb_spec slot move_from) as [-> | _]; [ | discriminate ].
      destruct (lookup move_from decls); simpl; [ | discriminate ].
      intros _. eauto.
Qed.

(* Two observations of one value that disagree. *)
Ltac observations_conflict_of :=
  match goal with
  | Hblock : block_of ?st ?v = Some _, H : closure_of ?st ?v = Some _ |- _ =>
    exfalso;
    destruct (FlambdaValueProofs.observations_disjoint st v) as (Hd & _);
    destruct Hd as (Hd & _); [ rewrite Hblock; discriminate | ];
    rewrite H in Hd; discriminate
  | Hblock : block_of ?st ?v = Some _, H : boxed_number_of ?st ?v = Some _
    |- _ =>
    exfalso;
    destruct (FlambdaValueProofs.observations_disjoint st v) as (Hd & _);
    destruct Hd as (_ & Hd & _); [ rewrite Hblock; discriminate | ];
    rewrite H in Hd; discriminate
  | Hblock : block_of ?st ?v = Some _, H : string_of ?st ?v = Some _ |- _ =>
    exfalso;
    destruct (FlambdaValueProofs.observations_disjoint st v) as (Hd & _);
    destruct Hd as (_ & _ & Hd); [ rewrite Hblock; discriminate | ];
    rewrite H in Hd; discriminate
  | Hclosure : closure_of ?st ?v = Some _, H : boxed_number_of ?st ?v = Some _
    |- _ =>
    exfalso;
    destruct (FlambdaValueProofs.observations_disjoint st v) as (_ & Hd & _);
    destruct Hd as (Hd & _); [ rewrite Hclosure; discriminate | ];
    rewrite H in Hd; discriminate
  | Hclosure : closure_of ?st ?v = Some _, H : string_of ?st ?v = Some _
    |- _ =>
    exfalso;
    destruct (FlambdaValueProofs.observations_disjoint st v) as (_ & Hd & _);
    destruct Hd as (_ & Hd); [ rewrite Hclosure; discriminate | ];
    rewrite H in Hd; discriminate
  | Hboxed : boxed_number_of ?st ?v = Some _, H : string_of ?st ?v = Some _
    |- _ =>
    exfalso;
    destruct (FlambdaValueProofs.observations_disjoint st v) as (_ & _ & Hd);
    rewrite H in Hd; specialize (Hd ltac:(rewrite Hboxed; discriminate));
    discriminate
  | Hnot : is_allocated ?st ?v = false, H : block_of ?st ?v = Some _ |- _ =>
    exfalso; destruct (not_allocated st v Hnot) as (_ & Hd & _);
    rewrite H in Hd; discriminate
  | Hnot : is_allocated ?st ?v = false, H : closure_of ?st ?v = Some _ |- _ =>
    exfalso; destruct (not_allocated st v Hnot) as (Hd & _);
    rewrite H in Hd; discriminate
  | Hnot : is_allocated ?st ?v = false, H : boxed_number_of ?st ?v = Some _
    |- _ =>
    exfalso; destruct (not_allocated st v Hnot) as (_ & _ & Hd & _);
    rewrite H in Hd; discriminate
  | Hnot : is_allocated ?st ?v = false, H : string_of ?st ?v = Some _ |- _ =>
    exfalso; destruct (not_allocated st v Hnot) as (_ & _ & _ & Hd);
    rewrite H in Hd; discriminate
  | Harray : array_of ?st ?v = Some _, H : closure_of ?st ?v = Some _ |- _ =>
    exfalso;
    destruct
      (FlambdaValueProofs.array_of_disjoint st v
         ltac:(rewrite Harray; discriminate))
      as (Hd & _);
    rewrite H in Hd; discriminate
  | Harray : array_of ?st ?v = Some _, H : block_of ?st ?v = Some _ |- _ =>
    exfalso;
    destruct
      (FlambdaValueProofs.array_of_disjoint st v
         ltac:(rewrite Harray; discriminate))
      as (_ & Hd & _);
    rewrite H in Hd; discriminate
  | Harray : array_of ?st ?v = Some _, H : boxed_number_of ?st ?v = Some _
    |- _ =>
    exfalso;
    destruct
      (FlambdaValueProofs.array_of_disjoint st v
         ltac:(rewrite Harray; discriminate))
      as (_ & _ & Hd & _);
    rewrite H in Hd; discriminate
  | Harray : array_of ?st ?v = Some _, H : string_of ?st ?v = Some _ |- _ =>
    exfalso;
    destruct
      (FlambdaValueProofs.array_of_disjoint st v
         ltac:(rewrite Harray; discriminate))
      as (_ & _ & _ & Hd);
    rewrite H in Hd; discriminate
  | Hnot : is_allocated ?st ?v = false, H : array_of ?st ?v = Some _ |- _ =>
    exfalso; rewrite (not_allocated_array st v Hnot) in H; discriminate
  end.

(* The same, where one observation is that of a closure of a set. *)
Ltac observations_conflict :=
  first
    [ observations_conflict_of
    | match goal with
      | H : closure_in_set ?st ?v = Some (_, _, _) |- _ =>
        let code := fresh "code" in
        let Hcode := fresh "Hcode" in
        destruct (closure_in_set_closure_of _ _ _ _ _ H)
          as (code & _ & Hcode);
        observations_conflict_of
      end ].

(* Two values with a head in common are of one kind. *)
Lemma block_of_kind (statics : static_env) (v : value) (b : Z * list value) :
  block_of statics v = Some b -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma boxed_number_of_kind (statics : static_env) (v : value) (c : const) :
  boxed_number_of statics v = Some c -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma string_of_kind (statics : static_env) (v : value) (bytes : list Z) :
  string_of statics v = Some bytes -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma array_of_kind (statics : static_env) (v : value)
    (elements : list value) :
  array_of statics v = Some elements -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma closure_of_kind (statics : static_env) (v : value)
    (c : code_id * list (value_slot * value)) :
  closure_of statics v = Some c -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma closure_in_set_kind (statics : static_env) (v : value)
    (c :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value)) :
  closure_in_set statics v = Some c -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; reflexivity. Qed.

Lemma value_step_kind (poison_refines : bool) (r : correspondence)
    (statics statics' : static_env) (R : value -> value -> Prop)
    (v v' : value) :
  value_step poison_refines r statics statics' R v v' ->
  kind_of_value v = kind_of_value v'.
Proof.
  intros Hstep.
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements]; try reflexivity.
  - rewrite (block_of_kind _ _ _ Hv), (block_of_kind _ _ _ Hv'). reflexivity.
  - rewrite (boxed_number_of_kind _ _ _ Hv), (boxed_number_of_kind _ _ _ Hv').
    reflexivity.
  - rewrite (string_of_kind _ _ _ Hv), (string_of_kind _ _ _ Hv'). reflexivity.
  - rewrite (closure_in_set_kind _ _ _ Hv), (closure_in_set_kind _ _ _ Hv').
    reflexivity.
  - simpl. symmetry. exact Hkind.
  - rewrite (array_of_kind _ _ _ Hv), (array_of_kind _ _ _ Hv'). reflexivity.
Qed.

(* Where the relation on contents holds of every two values that have
   their heads in common, the closures at the other function slots of two
   such values are related. *)
Lemma value_step_project (poison_refines : bool) (r : correspondence)
    (st st' : static_env) (R : value -> value -> Prop) (v v' w : value)
    (move_from move_to : function_slot) :
  (∀ a a' : value, value_step poison_refines r st st' R a a' -> R a a') ->
  value_step poison_refines r st st' R v v' ->
  project_function_slot st move_from move_to v = Some w ->
  ∃ w' : value,
    project_function_slot st' move_from move_to v' = Some w' ∧ R w w'.
Proof.
  intros Hclosed Hstep Hw.
  destruct (project_function_slot_closure _ _ _ _ _ Hw)
    as (decls & slots & Hclosure).
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots0 slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements];
    try discriminate;
    try observations_conflict.
  - rewrite Hv in Hclosure. injection Hclosure as <- <- <-.
    destruct Hsiblings as [[Hbound Hbound'] | [Hsiblings _]];
      [ | exact (Hsiblings _ _ Hw) ].
    destruct
      (project_set_bound_both _ _ _ _ _ _ _ _ _ _ Hbound Hbound' Hv Hv' Hw)
      as (w' & Hw' & Hbounds & Hw_closure & Hw'_closure).
    exists w'. split; [ exact Hw' | ]. apply Hclosed.
    exact
      (Step_closure _ _ _ _ _ _ _ move_to fdecls slots0 slots' Hw_closure
         Hw'_closure Hnames Hslots (or_introl Hbounds)).
  - simpl in Hclosure. rewrite Hgslot in Hclosure. discriminate.
Qed.

Theorem related_projections : related_projections_statement.
Proof.
  intros poison_refines r statics statics' h h' v v' w move_from move_to
    Hvalid Hrelated Hw.
  apply (value_step_project poison_refines r statics statics' _ v v');
    [ intros a a' Hstep; apply Related_step; exact Hstep
    | exact (value_related_step _ _ _ _ _ _ _ _ Hvalid Hrelated)
    | exact Hw ].
Qed.

Lemma value_avoids_nothing (v : value) : value_avoids nothing_dropped v = true.
Proof.
  induction v
    as [c | code slots IH | fslot decls slots IH | s | tag fields IH
       | elements IH | c | l
       | callee0 applied applied_arity IHcallee IHapplied]
    using value_induction; try reflexivity; simpl.
  - apply forallb_forall. intros [name v] Hin. rewrite Forall_forall in IH.
    exact (IH _ Hin).
  - apply andb_true_iff. split.
    + apply forallb_forall. intros decl _. reflexivity.
    + apply forallb_forall. intros [name v] Hin.
      rewrite Forall_forall in IH. exact (IH _ Hin).
  - apply forallb_forall. intros v Hin. rewrite Forall_forall in IH.
    exact (IH _ Hin).
  - apply forallb_forall. intros v Hin. rewrite Forall_forall in IH.
    exact (IH _ Hin).
  - rewrite IHcallee. simpl. apply forallb_forall. intros v Hin.
    rewrite Forall_forall in IHapplied. exact (IHapplied _ Hin).
Qed.

(* A value that refers to no symbol left out is related to itself, under
   any two static environments, by a correspondence that relates each of
   its locations to itself and every symbol that is not left out to
   itself. *)
Lemma value_related_same (D : dropped) (poison_refines : bool)
    (r : correspondence) (st st1 : static_env) (v : value) :
  (∀ s : symbol,
    symbol_kept D s = true ->
    corr_values r (Value_symbol s) (Value_symbol s)) ->
  value_avoids D v = true ->
  (∀ l : location, l ∈ locations_of_value v -> corr_locations r l l) ->
  value_related poison_refines r st st1 v v.
Proof.
  intros Hsymbols.
  induction v
    as [c | code slots IH | fslot decls slots IH | s | tag fields IH
       | elements IH | c | l
       | callee0 applied applied_arity IHcallee IHapplied]
    using value_induction; intros Havoids Hlocations.
  - apply related_const.
  - simpl in Havoids. apply andb_true_iff in Havoids as [_ Havoids].
    apply
      (related_closure _ _ _ _ _ _ unnamed_function_slot
         [(unnamed_function_slot, code)] slots slots);
      [ reflexivity | reflexivity | reflexivity | | left; split; exact I ].
    apply Forall2_map_same. intros slot Hslot.
    rewrite Forall_forall in IH. apply (IH slot Hslot).
    + rewrite forallb_forall in Havoids. specialize (Havoids slot Hslot).
      destruct slot. exact Havoids.
    + intros l Hl. apply Hlocations. simpl. apply in_flat_map.
      exists slot. auto.
  - simpl in Havoids. apply andb_true_iff in Havoids as [_ Havoids].
    destruct (lookup fslot decls) as [code | ] eqn:Hfslot;
      [ | apply Related_step, Step_same_closure_of_set; exact Hfslot ].
    apply (related_closure _ _ _ _ _ _ fslot decls slots slots);
      [ simpl; rewrite Hfslot; reflexivity
      | simpl; rewrite Hfslot; reflexivity | reflexivity | | ].
    + apply Forall2_map_same. intros slot Hslot.
      rewrite Forall_forall in IH. apply (IH slot Hslot).
      * rewrite forallb_forall in Havoids. specialize (Havoids slot Hslot).
        destruct slot. exact Havoids.
      * intros l Hl. apply Hlocations. simpl. apply in_flat_map.
        exists slot. auto.
    + left. split; exact I.
  - apply Related_corr. apply Hsymbols. exact Havoids.
  - apply (related_block _ _ _ _ _ _ tag fields fields);
      [ reflexivity | reflexivity | ].
    apply Forall2_same. intros field Hfield.
    rewrite Forall_forall in IH. apply (IH field Hfield).
    + simpl in Havoids. rewrite forallb_forall in Havoids.
      exact (Havoids field Hfield).
    + intros l Hl. apply Hlocations. simpl. apply in_flat_map.
      exists field. auto.
  - apply Related_step.
    apply (Step_array _ _ _ _ _ _ _ elements elements);
      [ reflexivity | reflexivity | ].
    apply Forall2_same. intros element Helement.
    rewrite Forall_forall in IH. apply (IH element Helement).
    + simpl in Havoids. rewrite forallb_forall in Havoids.
      exact (Havoids element Helement).
    + intros l Hl. apply Hlocations. simpl. apply in_flat_map.
      exists element. auto.
  - apply Related_step, Step_same_boxed_number.
  - apply related_location. apply Hlocations. left. reflexivity.
  - simpl in Havoids. apply andb_true_iff in Havoids as [Hcallee Hargs].
    apply Related_step, Step_partial_application.
    + apply IHcallee; [ exact Hcallee | ].
      intros l Hl. apply Hlocations. simpl. apply in_or_app. left. exact Hl.
    + apply Forall2_same. intros arg Harg.
      rewrite Forall_forall in IHapplied. apply (IHapplied arg Harg).
      * rewrite forallb_forall in Hargs. exact (Hargs arg Harg).
      * intros l Hl. apply Hlocations. simpl. apply in_or_app. right.
        apply in_flat_map. exists arg. auto.
Qed.

(* Cells related with one relation between values are related with
   another that contains it. *)
Lemma cell_related_change (poison_refines poison_refines1 : bool)
    (r r1 : correspondence) (st st' st1 st1' : static_env) (c c' : cell) :
  (∀ v v' : value,
    value_related poison_refines r st st' v v' ->
    value_related poison_refines1 r1 st1 st1' v v') ->
  cell_related poison_refines r st st' c c' ->
  cell_related poison_refines1 r1 st1 st1' c c'.
Proof.
  intros Hmono.
  destruct c as [m t fs | k es], c' as [m' t' fs' | k' es']; simpl;
    try exact (fun H => H).
  - intros (Hm & Ht & Hfs). split; [ exact Hm | ]. split; [ exact Ht | ].
    revert Hfs. apply Forall2_mono. exact Hmono.
  - intros (Hk & Hes). split; [ exact Hk | ].
    revert Hes. apply Forall2_mono. exact Hmono.
Qed.

Theorem outcome_related_reflexive : outcome_related_reflexive_statement.
Proof.
  intros poison_refines [vs statics h] [[Hheap Hstatics] Hvalues].
  simpl in Hheap, Hstatics, Hvalues.
  pose (r :=
          Correspondence (fun l l' => l = l' ∧ is_bound l h = true)
            (fun v v' => ∃ s : symbol, v = Value_symbol s ∧ v' = v)).
  assert (Hsame :
            ∀ v : value,
              value_closed h v ->
              value_related poison_refines r statics statics v v).
  { intros v Hclosed. apply (value_related_same nothing_dropped).
    - intros s _. exists s. auto.
    - apply value_avoids_nothing.
    - intros l Hl. split; [ reflexivity | exact (Hclosed l Hl) ]. }
  exists r. split; [ split; [ | split ] | ]; simpl.
  - intros a a' b b' [-> _] [-> _]. reflexivity.
  - intros l l' [<- Hbound]. unfold is_bound in Hbound.
    destruct (lookup l h) as [c | ] eqn:Hlookup; [ | discriminate ].
    exists c, c. split; [ reflexivity | ]. split; [ reflexivity | ].
    destruct c as [mut tag fields | kind elements]; simpl.
    + split; [ reflexivity | ]. split; [ reflexivity | ].
      apply Forall2_same. intros field Hfield. apply Hsame.
      exact (proj1 (Hheap l _ field Hlookup Hfield)).
    + split; [ reflexivity | ].
      apply Forall2_same. intros element Helement. apply Hsame.
      exact (proj1 (Hheap l _ element Hlookup Helement)).
  - intros v v' (s & -> & ->).
    destruct (lookup s (static_values statics)) as [a | ] eqn:Hlookup.
    + pose proof (Hstatics s a) as Hclosed.
      destruct a
        as [code slots | slot syms decls slots | tag fields | contents
           | bytes | elements | ].
      * apply
          (Step_closure _ _ _ _ _ _ _ unnamed_function_slot
             [(unnamed_function_slot, code)] slots slots);
          [ simpl; rewrite Hlookup; reflexivity
          | simpl; rewrite Hlookup; reflexivity | reflexivity
          | | left; split; unfold set_bound; rewrite Hlookup; exact I ].
        apply Forall2_same. intros v Hv. apply Hsame.
        exact (Hclosed v Hlookup Hv).
      * destruct (lookup slot decls) as [code | ] eqn:Hslot.
        -- apply (Step_closure _ _ _ _ _ _ _ slot decls slots slots);
             [ simpl; rewrite Hlookup; simpl; rewrite Hslot; reflexivity
             | simpl; rewrite Hlookup; simpl; rewrite Hslot; reflexivity
             | reflexivity | | ].
           ++ apply Forall2_same. intros v Hv. apply Hsame.
              exact (Hclosed v Hlookup Hv).
           ++ right. split; [ | intros move_to Hnone; exact Hnone ].
              intros move_to w Hw. exists w. split; [ exact Hw | ].
              destruct (project_symbol _ _ _ _ _ Hw) as (t & ->).
              apply Related_corr. exists t. auto.
        -- apply Step_opaque_symbol; unfold is_allocated; simpl;
             rewrite Hlookup, Hslot; reflexivity.
      * apply (Step_block _ _ _ _ _ _ _ tag fields fields);
          simpl; try rewrite Hlookup; try reflexivity.
        apply Forall2_same. intros v Hv. apply Hsame.
        exact (Hclosed v Hlookup Hv).
      * destruct (boxed_number_of statics (Value_symbol s)) as [c | ]
          eqn:Hboxed.
        -- exact
             (Step_boxed_number _ _ _ _ _ _ _ c c Hboxed Hboxed
                (or_introl eq_refl)).
        -- assert (Hnot : is_allocated statics (Value_symbol s) = false).
           { unfold is_allocated. rewrite Hboxed. simpl. rewrite Hlookup.
             reflexivity. }
           apply Step_opaque_symbol; exact Hnot.
      * apply (Step_string _ _ _ _ _ _ _ bytes); simpl; rewrite Hlookup;
          reflexivity.
      * apply (Step_array _ _ _ _ _ _ _ elements elements);
          simpl; try rewrite Hlookup; try reflexivity.
        apply Forall2_same. intros v Hv. apply Hsame.
        exact (Hclosed v Hlookup Hv).
      * apply Step_opaque_symbol; unfold is_allocated; simpl;
          rewrite Hlookup; reflexivity.
    + assert (Hnot : is_allocated statics (Value_symbol s) = false).
      { unfold is_allocated. simpl. rewrite Hlookup. reflexivity. }
      apply Step_opaque_symbol; exact Hnot.
  - apply Forall2_same. intros v Hv. apply Hsame. exact (Hvalues v Hv).
Qed.

(* The relation with poison refined contains the one without. *)
Lemma value_step_poison (r : correspondence) (statics statics' : static_env)
    (R : value -> value -> Prop) (v v' : value) :
  value_step false r statics statics' R v v' ->
  value_step true r statics statics' R v v'.
Proof.
  intros Hstep.
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements].
  - apply Step_const.
  - apply Step_location. exact Hl.
  - exact (Step_block _ _ _ _ _ _ _ tag fields fields' Hv Hv' Hfields).
  - destruct Hc as [-> | [Hfalse _]]; [ | discriminate Hfalse ].
    exact
      (Step_boxed_number _ _ _ _ _ _ _ c' c' Hv Hv' (or_introl eq_refl)).
  - apply Step_same_boxed_number.
  - exact (Step_string _ _ _ _ _ _ _ bytes Hv Hv').
  - exact
      (Step_closure _ _ _ _ _ _ _ fslot fdecls slots slots' Hv Hv' Hnames
         Hslots Hsiblings).
  - apply Step_same_closure_of_set. exact Hgslot.
  - apply Step_opaque_symbol; assumption.
  - discriminate Hpoison.
  - apply Step_partial_application; assumption.
  - exact (Step_array _ _ _ _ _ _ _ elements elements' Hv Hv' Helements).
Qed.

Lemma value_related_poison (r : correspondence)
    (statics statics' : static_env) (v v' : value) :
  value_related false r statics statics' v v' ->
  value_related true r statics statics' v v'.
Proof.
  revert v v'. apply value_related_induction.
  - intros v v' Hcorr. apply Related_corr. exact Hcorr.
  - intros v v' Hstep. apply Related_step. apply value_step_poison.
    revert Hstep. apply value_step_mono. intros a a' [_ Ha]. exact Ha.
Qed.

Lemma values_related_poison (r : correspondence)
    (statics statics' : static_env) (vs vs' : list value) :
  Forall2 (value_related false r statics statics') vs vs' ->
  Forall2 (value_related true r statics statics') vs vs'.
Proof. apply Forall2_mono. apply value_related_poison. Qed.

Theorem outcome_equivalent_refines : outcome_equivalent_refines_statement.
Proof.
  intros o o' (r & (Hbijection & Hcells & Hsymbols) & Hvalues).
  exists r. split; [ split; [ exact Hbijection | split ] | ].
  - intros l l' Hl. destruct (Hcells l l' Hl) as (c & c' & Hc & Hc' & Hrelated).
    exists c, c'. split; [ exact Hc | ]. split; [ exact Hc' | ].
    destruct c as [mut tag fields | kind elements],
      c' as [mut' tag' fields' | kind' elements'];
      try (exfalso; exact Hrelated).
    + destruct Hrelated as (Hmut & Htag & Hfields).
      split; [ exact Hmut | ]. split; [ exact Htag | ].
      apply values_related_poison. exact Hfields.
    + destruct Hrelated as (Hkind & Helements).
      split; [ exact Hkind | ].
      apply values_related_poison. exact Helements.
  - intros v v' Hv. apply value_step_poison.
    generalize (Hsymbols v v' Hv). apply value_step_mono.
    apply value_related_poison.
  - apply values_related_poison. exact Hvalues.
Qed.

(* What a location, and what a tagged immediate, may be related to. *)
Lemma value_related_location (poison_refines : bool) (r : correspondence)
    (statics statics' : static_env) (h h' : heap) (l : location)
    (v' : value) :
  correspondence_valid poison_refines r statics statics' h h' ->
  value_related poison_refines r statics statics' (Value_location l) v' ->
  ∃ l' : location, v' = Value_location l' ∧ corr_locations r l l'.
Proof.
  intros Hvalid Hrelated.
  pose proof (value_related_step _ _ _ _ _ _ _ _ Hvalid Hrelated) as Hstep.
  inversion Hstep; subst; try discriminate.
  eexists. split; [ reflexivity | assumption ].
Qed.

Lemma value_related_immediate (poison_refines : bool) (r : correspondence)
    (statics statics' : static_env) (h h' : heap) (i : Z) (v' : value) :
  correspondence_valid poison_refines r statics statics' h h' ->
  value_related poison_refines r statics statics'
    (Value_const (Tagged_immediate i)) v' ->
  v' = Value_const (Tagged_immediate i).
Proof.
  intros Hvalid Hrelated.
  pose proof (value_related_step _ _ _ _ _ _ _ _ Hvalid Hrelated) as Hstep.
  inversion Hstep; subst; try discriminate.
  reflexivity.
Qed.

Theorem aliasing_observed : aliasing_observed_statement.
Proof.
  intros l1 l2 l c Hdistinct (r & Hvalid & Hvalues).
  simpl in Hvalues.
  inversion Hvalues as [ | v1 v1' vs vs' Hfirst Hrest]; subst.
  inversion Hrest as [ | v2 v2' vs2 vs2' Hsecond _]; subst.
  apply (value_related_location _ _ _ _ _ _ _ _ Hvalid) in Hfirst
    as (l' & [= <-] & Hfirst).
  apply (value_related_location _ _ _ _ _ _ _ _ Hvalid) in Hsecond
    as (l'' & [= <-] & Hsecond).
  destruct Hvalid as (Hbijection & _).
  apply Hdistinct. apply (Hbijection l1 l2 l l Hfirst Hsecond). reflexivity.
Qed.

Theorem location_names_not_observed : location_names_not_observed_statement.
Proof.
  intros l l' c Hfields.
  exists (Correspondence (fun a b => a = l ∧ b = l') (fun _ _ => False)).
  split; [ split; [ | split ] | ]; simpl.
  - intros a a' b b' [-> ->] [-> ->]. split; reflexivity.
  - intros a b [-> ->]. rewrite !Pos.eqb_refl.
    eexists. eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
    destruct c as [mut tag fields | kind elements]; simpl in Hfields |- *;
      subst.
    + split; [ reflexivity | ]. split; [ reflexivity | ].
      constructor; [ apply related_const | constructor ].
    + split; [ reflexivity | ].
      constructor; [ apply related_const | constructor ].
  - intros v v' [ ].
  - constructor; [ | constructor ]. apply related_location. simpl. auto.
Qed.

Theorem block_and_symbol_related : block_and_symbol_related_statement.
Proof.
  intros s tag i.
  exists (Correspondence (fun _ _ => False) (fun _ _ => False)).
  split; [ split; [ | split ] | ]; simpl.
  - intros a a' b b' [ ].
  - intros l l' [ ].
  - intros v v' [ ].
  - constructor; [ | constructor ].
    apply
      (related_block _ _ _ _ _ _ tag [Value_const (Tagged_immediate i)]
         [Value_const (Tagged_immediate i)]).
    + reflexivity.
    + simpl. rewrite Pos.eqb_refl. reflexivity.
    + constructor; [ apply related_const | constructor ].
Qed.

Theorem garbage_not_observed : garbage_not_observed_statement.
Proof.
  intros l c vs [_ Hvalues]. simpl in Hvalues.
  exists
    (Correspondence (fun _ _ => False)
       (fun v v' => ∃ s : symbol, v = Value_symbol s ∧ v' = v)).
  split; [ split; [ | split ] | ]; simpl.
  - intros a a' b b' [ ].
  - intros l0 l0' [ ].
  - intros v v' (s & -> & ->). apply Step_opaque_symbol; reflexivity.
  - apply Forall2_same. intros v Hv.
    apply (value_related_same nothing_dropped).
    + intros s _. exists s. auto.
    + apply value_avoids_nothing.
    + intros l0 Hl0. specialize (Hvalues v Hv l0 Hl0). discriminate.
Qed.

Theorem boxed_poison_refined : boxed_poison_refined_statement.
Proof.
  intros s c Hc.
  assert (Hpoison :
            boxable_contents (Poison (kind_of_const c))
            = Some (Poison (kind_of_const c))).
  { unfold boxable_contents, boxable_number_of_const in Hc |- *. simpl.
    destruct (boxable_number_of_kind (kind_of_const c));
      [ reflexivity | discriminate ]. }
  assert (Hcontents : contents_related true (Poison (kind_of_const c)) c).
  { right. split; [ reflexivity | ]. exists (kind_of_const c). auto. }
  split; exists (Correspondence (fun _ _ => False) (fun _ _ => False));
    (split; [ split; [ | split ] | ]); simpl;
    try (intros a a' b b' [ ]); try (intros l l' [ ]);
    (constructor; [ | constructor ]); apply Related_step;
    apply (Step_boxed_number _ _ _ _ _ _ _ (Poison (kind_of_const c)) c);
    simpl; rewrite ?Pos.eqb_refl; assumption.
Qed.

(* ======================================================================== *)
(* The relation composes                                                    *)
(* ======================================================================== *)

(* Without poison the relation is symmetric. *)
Definition swap_correspondence (r : correspondence) : correspondence :=
  Correspondence (fun l' l => corr_locations r l l')
    (fun v' v => corr_values r v v').

Lemma Forall2_swap {A B : Type} (R : A -> B -> Prop) (R' : B -> A -> Prop)
    (l : list A) (l' : list B) :
  (∀ (a : A) (b : B), R a b -> R' b a) -> Forall2 R l l' -> Forall2 R' l' l.
Proof.
  intros Hswap Hall. induction Hall; constructor; auto.
Qed.

Lemma value_step_swap (r : correspondence) (st st' : static_env)
    (R R' : value -> value -> Prop) (v v' : value) :
  (∀ a a' : value, R a a' -> R' a' a) ->
  value_step false r st st' R v v' ->
  value_step false (swap_correspondence r) st' st R' v' v.
Proof.
  intros Hswap Hstep.
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements].
  - apply Step_const.
  - apply Step_location. exact Hl.
  - exact
      (Step_block _ _ _ _ _ _ _ tag fields' fields Hv' Hv
         (Forall2_swap _ _ _ _ Hswap Hfields)).
  - destruct Hc as [-> | [Hfalse _]]; [ | discriminate Hfalse ].
    exact
      (Step_boxed_number _ _ _ _ _ _ _ c' c' Hv' Hv (or_introl eq_refl)).
  - apply Step_same_boxed_number.
  - exact (Step_string _ _ _ _ _ _ _ bytes Hv' Hv).
  - apply
      (Step_closure _ _ _ _ _ _ _ fslot fdecls slots' slots Hv' Hv
         (eq_sym Hnames) (Forall2_swap _ _ _ _ Hswap Hslots)).
    destruct Hsiblings as [[Hbound Hbound'] | [Hsiblings Hnone]];
      [ left; split; assumption | ].
    right. split.
    + intros move_to w' Hw'.
      destruct (project_function_slot st fslot move_to v) as [w | ] eqn:Hw.
      * destruct (Hsiblings _ _ Hw) as (w0 & Hw0 & Hrelated).
        rewrite Hw' in Hw0. injection Hw0 as <-.
        exists w. split; [ reflexivity | exact (Hswap _ _ Hrelated) ].
      * rewrite (Hnone _ Hw) in Hw'. discriminate.
    + intros move_to Hw'.
      destruct (project_function_slot st fslot move_to v) as [w | ] eqn:Hw;
        [ | reflexivity ].
      destruct (Hsiblings _ _ Hw) as (w0 & Hw0 & _).
      rewrite Hw' in Hw0. discriminate.
  - apply Step_same_closure_of_set. exact Hgslot.
  - apply Step_opaque_symbol; assumption.
  - discriminate Hpoison.
  - apply Step_partial_application;
      [ exact (Hswap _ _ Hpcallee)
      | exact (Forall2_swap _ _ _ _ Hswap Hpargs) ].
  - exact
      (Step_array _ _ _ _ _ _ _ elements' elements Hv' Hv
         (Forall2_swap _ _ _ _ Hswap Helements)).
Qed.

Lemma value_related_swap (r : correspondence) (st st' : static_env)
    (v v' : value) :
  value_related false r st st' v v' ->
  value_related false (swap_correspondence r) st' st v' v.
Proof.
  revert v v'. apply value_related_induction.
  - intros v v' Hcorr. apply Related_corr. exact Hcorr.
  - intros v v' Hstep. apply Related_step. revert Hstep.
    apply value_step_swap. intros a a' [_ Ha]. exact Ha.
Qed.

Theorem outcome_equivalent_symmetric : outcome_equivalent_symmetric_statement.
Proof.
  intros [vs statics h] [vs' statics' h']
    (r & (Hbijection & Hcells & Hvalues) & Hexit).
  simpl in Hbijection, Hcells, Hvalues, Hexit.
  exists (swap_correspondence r). simpl.
  split; [ split; [ | split ] | ].
  - intros a a' b b' Hab Hab'. simpl in Hab, Hab'.
    destruct (Hbijection _ _ _ _ Hab Hab') as [Hto Hfrom].
    split; assumption.
  - intros l' l Hl. simpl in Hl.
    destruct (Hcells _ _ Hl) as (c & c' & Hc & Hc' & Hcell).
    exists c', c. split; [ exact Hc' | ]. split; [ exact Hc | ].
    destruct c as [m t fs | k es], c' as [m' t' fs' | k' es'];
      try (exfalso; exact Hcell).
    + destruct Hcell as (Hm & Ht & Hfs).
      split; [ exact (eq_sym Hm) | ]. split; [ exact (eq_sym Ht) | ].
      exact
        (Forall2_swap _ _ _ _ (value_related_swap r statics statics') Hfs).
    + destruct Hcell as (Hk & Hes). split; [ exact (eq_sym Hk) | ].
      exact
        (Forall2_swap _ _ _ _ (value_related_swap r statics statics') Hes).
  - intros v' v Hv. simpl in Hv.
    exact
      (value_step_swap r statics statics' _ _ v v'
         (value_related_swap r statics statics') (Hvalues _ _ Hv)).
  - exact (Forall2_swap _ _ _ _ (value_related_swap r statics statics') Hexit).
Qed.

Lemma contents_related_compose (poison_refines : bool) (c1 c2 c3 : const) :
  contents_related poison_refines c1 c2 ->
  contents_related poison_refines c2 c3 ->
  contents_related poison_refines c1 c3.
Proof.
  intros [-> | (Hflag & kind & -> & Hkind)] Hsecond; [ exact Hsecond | ].
  right. split; [ exact Hflag | ]. exists kind. split; [ reflexivity | ].
  destruct Hsecond as [<- | (_ & kind' & -> & Hkind')]; [ exact Hkind | ].
  simpl in Hkind. subst kind'. exact Hkind'.
Qed.

(* Two steps that share their middle value make one, on the composition
   of what relates their contents. *)
Lemma value_step_compose (poison_refines : bool) (r1 r2 r : correspondence)
    (statics1 statics2 statics3 : static_env) (R1 R2 : value -> value -> Prop)
    (v1 v2 v3 : value) :
  (∀ l1 l2 l3 : location,
    corr_locations r1 l1 l2 -> corr_locations r2 l2 l3 ->
    corr_locations r l1 l3) ->
  (∀ a b : value,
    value_step poison_refines r1 statics1 statics2 R1 a b -> R1 a b) ->
  (∀ b c : value,
    value_step poison_refines r2 statics2 statics3 R2 b c -> R2 b c) ->
  value_step poison_refines r1 statics1 statics2 R1 v1 v2 ->
  value_step poison_refines r2 statics2 statics3 R2 v2 v3 ->
  value_step poison_refines r statics1 statics3
    (fun a c => ∃ b : value, R1 a b ∧ R2 b c) v1 v3.
Proof.
  intros Hlocations Hclosed1 Hclosed2 Hfirst Hsecond.
  destruct Hfirst
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements].
  - (* a constant *)
    inversion Hsecond; subst; simpl in *; try discriminate.
    + apply Step_const.
    + apply Step_poison; reflexivity.
  - (* a location *)
    inversion Hsecond; subst; simpl in *; try discriminate.
    apply Step_location. eapply Hlocations; eassumption.
  - (* a block *)
    inversion Hsecond; subst; try observations_conflict; simpl in *;
      try discriminate.
    match goal with
    | H : block_of statics2 v' = Some _ |- _ =>
      rewrite Hv' in H; injection H as <- <-
    end.
    eapply Step_block; [ exact Hv | eassumption | ].
    eapply Forall2_compose; eassumption.
  - (* a boxed number *)
    inversion Hsecond; subst; try observations_conflict; simpl in *;
      try discriminate.
    + match goal with
      | H : boxed_number_of statics2 v' = Some _ |- _ =>
        rewrite Hv' in H; injection H as <-
      end.
      eapply Step_boxed_number; [ exact Hv | eassumption | ].
      eapply contents_related_compose; eassumption.
    + apply boxable_contents_same in Hv' as Hsame. subst.
      eapply Step_boxed_number; [ exact Hv | exact Hv' | exact Hc ].
  - (* the same boxed number *)
    inversion Hsecond; subst; simpl in *; try discriminate.
    + match goal with
      | H : boxable_contents c = Some _ |- _ =>
        apply boxable_contents_same in H as Hsame; subst;
        eapply Step_boxed_number; [ exact H | eassumption | eassumption ]
      end.
    + apply Step_same_boxed_number.
  - (* a string *)
    inversion Hsecond; subst; try observations_conflict; simpl in *;
      try discriminate.
    match goal with
    | H : string_of statics2 v' = Some _ |- _ =>
      rewrite Hv' in H; injection H as <-
    end.
    eapply Step_string; eassumption.
  - (* a closure *)
    destruct Hsecond
      as [c | l l' Hl | v2 v3 tag fields fields' Hv2 Hv3 Hfields
         | v2 v3 c c' Hv2 Hv3 Hc | c | v2 v3 bytes Hv2 Hv3
         | v2 v3 fslot2 fdecls2 slots2 slots3 Hv2 Hv3 Hnames2 Hslots2
           Hsiblings2
         | gslot gdecls gslots Hgslot | s Hs Hs'
         | kind v3 Hpoison Hkind
         | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
         | v2 v3 elements elements' Hv2 Hv3 Helements];
      try discriminate;
      try observations_conflict.
    + rewrite Hv' in Hv2. injection Hv2 as <- <- <-.
      apply
        (Step_closure _ _ _ _ _ _ _ fslot fdecls slots slots3 Hv Hv3);
        [ congruence | exact (Forall2_compose _ _ _ _ _ Hslots Hslots2) | ].
      destruct Hsiblings as [[Hbound1 Hbound2] | [Hsiblings Hnone]],
        Hsiblings2 as [[Hbound2' Hbound3] | [Hsiblings2 Hnone2]].
      * left. split; assumption.
      * right. split.
        2: { intros move_to Hw.
             apply Hnone2.
             apply
               (proj1 (project_set_bound _ _ _ _ _ move_to Hbound2 Hv')).
             apply
               (proj1 (project_set_bound _ _ _ _ _ move_to Hbound1 Hv)).
             exact Hw. }
        intros move_to w Hw.
        destruct
          (project_set_bound_both _ _ _ _ _ _ _ _ _ _ Hbound1 Hbound2 Hv Hv'
             Hw)
          as (w2 & Hw2 & Hbounds & Hw_closure & Hw2_closure).
        destruct (Hsiblings2 _ _ Hw2) as (w3 & Hw3 & Hrelated23).
        exists w3. split; [ exact Hw3 | ]. exists w2.
        split; [ | exact Hrelated23 ]. apply Hclosed1.
        exact
          (Step_closure _ _ _ _ _ _ _ move_to fdecls slots slots'
             Hw_closure Hw2_closure Hnames Hslots (or_introl Hbounds)).
      * right. split.
        2: { intros move_to Hw.
             apply
               (proj1 (project_set_bound _ _ _ _ _ move_to Hbound3 Hv3)).
             apply
               (proj1 (project_set_bound _ _ _ _ _ move_to Hbound2' Hv')).
             exact (Hnone _ Hw). }
        intros move_to w Hw.
        destruct (Hsiblings _ _ Hw) as (w2 & Hw2 & Hrelated12).
        destruct
          (project_set_bound_both _ _ _ _ _ _ _ _ _ _ Hbound2' Hbound3 Hv'
             Hv3 Hw2)
          as (w3 & Hw3 & Hbounds & Hw2_closure & Hw3_closure).
        exists w3. split; [ exact Hw3 | ]. exists w2.
        split; [ exact Hrelated12 | ]. apply Hclosed2.
        exact
          (Step_closure _ _ _ _ _ _ _ move_to fdecls slots' slots3
             Hw2_closure Hw3_closure Hnames2 Hslots2 (or_introl Hbounds)).
      * right. split; [ | intros move_to Hw; exact (Hnone2 _ (Hnone _ Hw)) ].
        intros move_to w Hw.
        destruct (Hsiblings _ _ Hw) as (w2 & Hw2 & Hrelated12).
        destruct (Hsiblings2 _ _ Hw2) as (w3 & Hw3 & Hrelated23).
        exists w3. split; [ exact Hw3 | ]. exists w2. auto.
    + simpl in Hv'. rewrite Hgslot in Hv'. discriminate.
  - (* a closure at a slot that its set has not got *)
    inversion Hsecond; subst; simpl in *; try discriminate.
    + rewrite Hgslot in *. discriminate.
    + apply Step_same_closure_of_set. exact Hgslot.
  - (* a symbol that is nothing *)
    inversion Hsecond; subst; try observations_conflict; simpl in *;
      try discriminate.
    apply Step_opaque_symbol; assumption.
  - (* poison *)
    apply Step_poison; [ exact Hpoison | ].
    rewrite <- (value_step_kind _ _ _ _ _ _ _ Hsecond). exact Hkind.
  - (* a partial application *)
    inversion Hsecond; subst; simpl in *; try discriminate.
    apply Step_partial_application;
      [ eexists; split; eassumption | eapply Forall2_compose; eassumption ].
  - (* an immutable array *)
    inversion Hsecond; subst; try observations_conflict; simpl in *;
      try discriminate.
    match goal with
    | H : array_of statics2 v' = Some _ |- _ =>
      rewrite Hv' in H; injection H as <-
    end.
    eapply Step_array; [ exact Hv | eassumption | ].
    eapply Forall2_compose; eassumption.
Qed.

Theorem outcome_related_transitive : outcome_related_transitive_statement.
Proof.
  intros poison_refines [vs1 statics1 h1] [vs2 statics2 h2] [vs3 statics3 h3]
    (r1 & Hvalid1 & Hvalues1) (r2 & Hvalid2 & Hvalues2).
  simpl in Hvalid1, Hvalues1, Hvalid2, Hvalues2.
  pose (r :=
          Correspondence
            (fun l1 l3 =>
               ∃ l2 : location,
                 corr_locations r1 l1 l2 ∧ corr_locations r2 l2 l3)
            (fun v1 v3 =>
               ∃ v2 : value,
                 value_related poison_refines r1 statics1 statics2 v1 v2
                 ∧ value_related poison_refines r2 statics2 statics3 v2 v3)).
  assert (Hcompose :
            ∀ vs vs' vs'' : list value,
              Forall2 (value_related poison_refines r1 statics1 statics2)
                vs vs' ->
              Forall2 (value_related poison_refines r2 statics2 statics3)
                vs' vs'' ->
              Forall2 (value_related poison_refines r statics1 statics3)
                vs vs'').
  { intros vs vs' vs'' Hfirst Hsecond.
    generalize (Forall2_compose _ _ _ _ _ Hfirst Hsecond).
    apply Forall2_mono. intros a c Hac. apply Related_corr. exact Hac. }
  exists r. simpl. split; [ split; [ | split ] | ].
  - destruct Hvalid1 as (Hbijection1 & _), Hvalid2 as (Hbijection2 & _).
    intros a a' c c' (b & Hab & Hbc) (b' & Hab' & Hbc'). simpl.
    rewrite (Hbijection1 a a' b b' Hab Hab').
    exact (Hbijection2 b b' c c' Hbc Hbc').
  - intros l1 l3 (l2 & Hl12 & Hl23).
    destruct Hvalid1 as (_ & Hcells1 & _), Hvalid2 as (_ & Hcells2 & _).
    destruct (Hcells1 _ _ Hl12) as (c1 & c2 & Hc1 & Hc2 & Hrelated12).
    destruct (Hcells2 _ _ Hl23) as (c2' & c3 & Hc2' & Hc3 & Hrelated23).
    rewrite Hc2 in Hc2'. injection Hc2' as <-.
    exists c1, c3. split; [ exact Hc1 | ]. split; [ exact Hc3 | ].
    destruct c1 as [mut1 tag1 fields1 | kind1 elements1],
      c2 as [mut2 tag2 fields2 | kind2 elements2],
      c3 as [mut3 tag3 fields3 | kind3 elements3];
      try (exfalso; exact Hrelated12); try (exfalso; exact Hrelated23).
    + destruct Hrelated12 as (-> & -> & Hfields12).
      destruct Hrelated23 as (-> & -> & Hfields23).
      split; [ reflexivity | ]. split; [ reflexivity | ].
      exact (Hcompose _ _ _ Hfields12 Hfields23).
    + destruct Hrelated12 as (-> & Helements12).
      destruct Hrelated23 as (-> & Helements23).
      split; [ reflexivity | ].
      exact (Hcompose _ _ _ Helements12 Helements23).
  - intros v1 v3 (v2 & Hv12 & Hv23).
    pose proof (value_related_step _ _ _ _ _ _ _ _ Hvalid1 Hv12) as Hstep12.
    pose proof (value_related_step _ _ _ _ _ _ _ _ Hvalid2 Hv23) as Hstep23.
    assert (Hlocations :
              ∀ l1 l2 l3 : location,
                corr_locations r1 l1 l2 -> corr_locations r2 l2 l3 ->
                corr_locations r l1 l3).
    { intros l1 l2 l3 H12 H23. exists l2. auto. }
    generalize
      (value_step_compose _ _ _ r _ _ _ _ _ _ _ _ Hlocations
         (fun a b Hstep => Related_step _ _ _ _ a b Hstep)
         (fun b c Hstep => Related_step _ _ _ _ b c Hstep) Hstep12 Hstep23).
    apply value_step_mono. intros a c Hac. apply Related_corr. exact Hac.
  - exact (Hcompose _ _ _ Hvalues1 Hvalues2).
Qed.

(* ======================================================================== *)
(* A write may not be deleted                                               *)
(* ======================================================================== *)

Theorem delete_write_wrong : delete_write_wrong_statement.
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
(* The machine respects the relation                                        *)
(* ======================================================================== *)

Definition const_of (v : value) : option const :=
  match v with
  | Value_const c => Some c
  | _ => None
  end.

Lemma int_of_value_const_of (kind : standard_int) (v : value) :
  int_of_value kind v
  = match const_of v with
    | Some c => int_of_value kind (Value_const c)
    | None => None
    end.
Proof. destruct v; try reflexivity; destruct kind; reflexivity. Qed.

Lemma is_location_location_of (v : value) :
  is_location v = is_some (location_of v).
Proof. destruct v; reflexivity. Qed.

Lemma eval_block_load_load_field (st : static_env) (kind : block_access_kind)
    (field : Z) (v : value) :
  eval_block_load st kind field v
  = let* '(tag, fields) := block_of st v in
    load_field kind field tag fields.
Proof. destruct kind. reflexivity. Qed.

Lemma Forall2_nth_error {A B : Type} (R : A -> B -> Prop) (l : list A)
    (l' : list B) (n : nat) (x : A) :
  Forall2 R l l' ->
  nth_error l n = Some x ->
  ∃ y : B, nth_error l' n = Some y ∧ R x y.
Proof.
  intros Hall. revert n.
  induction Hall as [ | a b l l' Hab Hall IH]; intros [ | n]; simpl;
    try discriminate.
  - intros [= <-]. exists b. auto.
  - apply IH.
Qed.

(* Two finite maps with the same names and related contents. *)
Lemma lookup_related {A : Type} (R : A -> A -> Prop)
    (l l' : list (positive * A)) (name : positive) (a : A) :
  map fst l = map fst l' ->
  Forall2 R (map snd l) (map snd l') ->
  lookup name l = Some a ->
  ∃ a' : A, lookup name l' = Some a' ∧ R a a'.
Proof.
  revert l'. induction l as [ | [n x] l IH]; intros [ | [n' x'] l']; simpl;
    try discriminate.
  intros [= <- Hnames] Hall.
  inversion Hall as [ | x0 y0 l0 l0' Hx Hrest]; subst.
  destruct (Pos.eqb name n).
  - intros [= <-]. exists x'. auto.
  - apply IH; assumption.
Qed.

(* A static environment that grows: what is bound stays bound, to the
   same thing. *)
Definition statics_extend (st st1 : static_env) : Prop :=
  ∀ (s : symbol) (a : statically_allocated_value),
    lookup s (static_values st) = Some a ->
    lookup s (static_values st1) = Some a.

Lemma block_of_extend (st st1 : static_env) (v : value)
    (b : Z * list value) :
  statics_extend st st1 -> block_of st v = Some b -> block_of st1 v = Some b.
Proof.
  intros Hextend. destruct v; simpl; try discriminate; auto.
  destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (Hextend _ _ Hlookup). auto.
Qed.

Lemma boxed_number_of_extend (st st1 : static_env) (v : value) (c : const) :
  statics_extend st st1 ->
  boxed_number_of st v = Some c ->
  boxed_number_of st1 v = Some c.
Proof.
  intros Hextend. destruct v; simpl; try discriminate; auto.
  destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (Hextend _ _ Hlookup). auto.
Qed.

Lemma string_of_extend (st st1 : static_env) (v : value) (bytes : list Z) :
  statics_extend st st1 ->
  string_of st v = Some bytes ->
  string_of st1 v = Some bytes.
Proof.
  intros Hextend. destruct v; simpl; try discriminate; auto.
  destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (Hextend _ _ Hlookup). auto.
Qed.

Lemma array_of_extend (st st1 : static_env) (v : value)
    (elements : list value) :
  statics_extend st st1 ->
  array_of st v = Some elements ->
  array_of st1 v = Some elements.
Proof.
  intros Hextend. destruct v; simpl; try discriminate; auto.
  destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (Hextend _ _ Hlookup). auto.
Qed.

Lemma closure_of_extend (st st1 : static_env) (v : value)
    (c : code_id * list (value_slot * value)) :
  statics_extend st st1 ->
  closure_of st v = Some c ->
  closure_of st1 v = Some c.
Proof.
  intros Hextend. destruct v; simpl; try discriminate; auto.
  destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (Hextend _ _ Hlookup). auto.
Qed.

Lemma closure_in_set_extend (st st1 : static_env) (v : value)
    (c :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value)) :
  statics_extend st st1 ->
  closure_in_set st v = Some c ->
  closure_in_set st1 v = Some c.
Proof.
  intros Hextend. destruct v; simpl; try discriminate; auto.
  destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (Hextend _ _ Hlookup). auto.
Qed.

(* The projection of a function slot from a closure does not change as
   the static environment grows. *)
Lemma project_function_slot_extend (st st1 : static_env) (v : value)
    (c :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value))
    (move_from move_to : function_slot) :
  statics_extend st st1 ->
  closure_in_set st v = Some c ->
  project_function_slot st1 move_from move_to v
  = project_function_slot st move_from move_to v.
Proof.
  intros Hextend. destruct v; simpl; try discriminate; auto.
  destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (Hextend _ _ Hlookup). auto.
Qed.

(* A set that is known is known as the static environment grows. *)
Lemma set_bound_extend (st st1 : static_env) (v : value)
    (c :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value)) :
  statics_extend st st1 ->
  closure_in_set st v = Some c ->
  set_bound st v ->
  set_bound st1 v.
Proof.
  intros Hextend. destruct v; simpl; try (intros _ _; exact I).
  destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup;
    [ | intros H; discriminate H ].
  rewrite (Hextend _ _ Hlookup). intros _.
  destruct a
    as [code slots | slot syms decls slots | tag fields | contents | bytes
        | elements | ];
    try (intros _; exact I).
  intros [Hkeys Hbound]. split; [ exact Hkeys | ].
  intros slot0 s0 Hin. exact (Hextend _ _ (Hbound slot0 s0 Hin)).
Qed.

(* The relation grows with the correspondence. *)
Lemma value_step_grow (poison_refines : bool) (r r1 : correspondence)
    (st st' : static_env) (R R1 : value -> value -> Prop) (v v' : value) :
  (∀ l l' : location, corr_locations r l l' -> corr_locations r1 l l') ->
  (∀ a a' : value, R a a' -> R1 a a') ->
  value_step poison_refines r st st' R v v' ->
  value_step poison_refines r1 st st' R1 v v'.
Proof.
  intros Hlocations Hmono Hstep.
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements].
  - apply Step_const.
  - apply Step_location. exact (Hlocations _ _ Hl).
  - exact
      (Step_block _ _ _ _ _ _ _ tag fields fields' Hv Hv'
         (Forall2_mono _ _ _ _ Hmono Hfields)).
  - exact (Step_boxed_number _ _ _ _ _ _ _ c c' Hv Hv' Hc).
  - apply Step_same_boxed_number.
  - exact (Step_string _ _ _ _ _ _ _ bytes Hv Hv').
  - apply
      (Step_closure _ _ _ _ _ _ _ fslot fdecls slots slots' Hv Hv' Hnames
         (Forall2_mono _ _ _ _ Hmono Hslots)).
    destruct Hsiblings as [Hbound | [Hsiblings Hnone]];
      [ left; exact Hbound | ].
    right. split; [ | exact Hnone ]. intros move_to w Hw.
    destruct (Hsiblings _ _ Hw) as (w' & Hw' & Hrelated).
    exists w'. split; [ exact Hw' | exact (Hmono _ _ Hrelated) ].
  - apply Step_same_closure_of_set. exact Hgslot.
  - apply Step_opaque_symbol; assumption.
  - apply Step_poison; assumption.
  - apply Step_partial_application;
      [ exact (Hmono _ _ Hpcallee)
      | exact (Forall2_mono _ _ _ _ Hmono Hpargs) ].
  - exact
      (Step_array _ _ _ _ _ _ _ elements elements' Hv Hv'
         (Forall2_mono _ _ _ _ Hmono Helements)).
Qed.

Lemma value_related_grow (poison_refines : bool) (r r1 : correspondence)
    (st st' : static_env) (v v' : value) :
  (∀ l l' : location, corr_locations r l l' -> corr_locations r1 l l') ->
  (∀ a a' : value, corr_values r a a' -> corr_values r1 a a') ->
  value_related poison_refines r st st' v v' ->
  value_related poison_refines r1 st st' v v'.
Proof.
  intros Hlocations Hvalues. revert v v'. apply value_related_induction.
  - intros v v' Hcorr. apply Related_corr. exact (Hvalues _ _ Hcorr).
  - intros v v' Hstep. apply Related_step. revert Hstep.
    apply value_step_grow; [ exact Hlocations | ].
    intros a a' [_ Ha]. exact Ha.
Qed.

(* The relation is kept when the two static environments grow, if every
   symbol that is nothing in both corresponds to itself: a symbol that
   was nothing and comes to be bound is still related to itself. *)
Lemma value_related_statics (poison_refines : bool) (r : correspondence)
    (st st' st1 st1' : static_env) (v v' : value) :
  statics_extend st st1 ->
  statics_extend st' st1' ->
  (∀ s : symbol,
    is_allocated st (Value_symbol s) = false ->
    is_allocated st' (Value_symbol s) = false ->
    corr_values r (Value_symbol s) (Value_symbol s)) ->
  value_related poison_refines r st st' v v' ->
  value_related poison_refines r st1 st1' v v'.
Proof.
  intros Hextend Hextend' Hsymbols. revert v v'.
  apply value_related_induction.
  - intros v v' Hcorr. apply Related_corr. exact Hcorr.
  - intros v v' Hstep.
    destruct Hstep
      as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
         | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
         | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
           Hsiblings
         | gslot gdecls gslots Hgslot | s Hs Hs'
         | kind v' Hpoison Hkind
         | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
         | v v' elements elements' Hv Hv' Helements].
    + apply related_const.
    + apply related_location. exact Hl.
    + apply (related_block _ _ _ _ _ _ tag fields fields');
        [ exact (block_of_extend _ _ _ _ Hextend Hv)
        | exact (block_of_extend _ _ _ _ Hextend' Hv')
        | ].
      revert Hfields. apply Forall2_mono. intros a a' [_ Ha]. exact Ha.
    + apply Related_step.
      exact
        (Step_boxed_number _ _ _ _ _ _ _ c c'
           (boxed_number_of_extend _ _ _ _ Hextend Hv)
           (boxed_number_of_extend _ _ _ _ Hextend' Hv') Hc).
    + apply Related_step, Step_same_boxed_number.
    + apply Related_step.
      exact
        (Step_string _ _ _ _ _ _ _ bytes
           (string_of_extend _ _ _ _ Hextend Hv)
           (string_of_extend _ _ _ _ Hextend' Hv')).
    + apply (related_closure _ _ _ _ _ _ fslot fdecls slots slots');
        [ exact (closure_in_set_extend _ _ _ _ Hextend Hv)
        | exact (closure_in_set_extend _ _ _ _ Hextend' Hv')
        | exact Hnames
        | | ].
      * revert Hslots. apply Forall2_mono. intros a a' [_ Ha]. exact Ha.
      * destruct Hsiblings as [[Hbound Hbound'] | [Hsiblings Hnone]].
        -- left.
           split;
             [ exact (set_bound_extend _ _ _ _ Hextend Hv Hbound)
             | exact (set_bound_extend _ _ _ _ Hextend' Hv' Hbound') ].
        -- right. split; intros move_to;
             rewrite (project_function_slot_extend _ _ _ _ _ _ Hextend Hv),
               (project_function_slot_extend _ _ _ _ _ _ Hextend' Hv');
             [ | apply Hnone ].
           intros w Hw.
           destruct (Hsiblings _ _ Hw) as (w' & Hw' & [_ Hrelated]).
           exists w'. split; [ exact Hw' | exact Hrelated ].
    + apply Related_step, Step_same_closure_of_set. exact Hgslot.
    + apply Related_corr. apply Hsymbols; assumption.
    + apply Related_step, Step_poison; assumption.
    + apply Related_step, Step_partial_application;
        [ exact (proj2 Hpcallee) | ].
      revert Hpargs. apply Forall2_mono. intros a a' [_ Ha]. exact Ha.
    + apply Related_step.
      apply
        (Step_array _ _ _ _ _ _ _ elements elements'
           (array_of_extend _ _ _ _ Hextend Hv)
           (array_of_extend _ _ _ _ Hextend' Hv')).
      revert Helements. apply Forall2_mono. intros a a' [_ Ha]. exact Ha.
Qed.

Section Machine.

Variable D : dropped.
Variable r : correspondence.
Variables statics statics' : static_env.
Variables h h' : heap.
Hypothesis Hmemories : memories_related D r statics statics' h h'.

#[local] Abbreviation related := (value_related false r statics statics').

Lemma memories_valid : correspondence_valid false r statics statics' h h'.
Proof. exact (proj1 Hmemories). Qed.

Lemma memories_names : names_related D r statics statics'.
Proof. exact (proj2 Hmemories). Qed.

Lemma memories_symbols (s : symbol) :
  symbol_kept D s = true -> corr_values r (Value_symbol s) (Value_symbol s).
Proof. exact (proj1 memories_names s). Qed.

Lemma memories_domain (s : symbol) :
  symbol_kept D s = true ->
  is_bound s (static_values statics) = is_bound s (static_values statics').
Proof. exact (proj1 (proj2 memories_names) s). Qed.

Lemma memories_left_out (s : symbol) :
  symbol_kept D s = false ->
  lookup s (static_values statics) = None
  ∧ is_allocated statics' (Value_symbol s) = true.
Proof. exact (proj1 (proj2 (proj2 memories_names)) s). Qed.

Lemma memories_imported (s : symbol) (c : import_class) :
  lookup s (static_values statics) = Some (Static_imported c)
  <-> lookup s (static_values statics') = Some (Static_imported c).
Proof. exact (proj2 (proj2 (proj2 memories_names)) s c). Qed.

(* A symbol that is nothing in both memories corresponds to itself. *)
Lemma memories_opaque (s : symbol) :
  is_allocated statics (Value_symbol s) = false ->
  is_allocated statics' (Value_symbol s) = false ->
  corr_values r (Value_symbol s) (Value_symbol s).
Proof.
  intros _ Hnot'. destruct (symbol_kept D s) eqn:Hkept.
  - exact (memories_symbols s Hkept).
  - destruct (memories_left_out s Hkept) as [_ Hallocated].
    rewrite Hallocated in Hnot'. discriminate.
Qed.

(* A symbol that the first memory binds, the second binds. *)
Lemma memories_bound (s : symbol) :
  is_bound s (static_values statics) = true ->
  is_bound s (static_values statics') = true.
Proof.
  intros Hbound. destruct (symbol_kept D s) eqn:Hkept.
  - rewrite <- (memories_domain s Hkept). exact Hbound.
  - destruct (memories_left_out s Hkept) as [Hnone _].
    unfold is_bound in Hbound. rewrite Hnone in Hbound. discriminate.
Qed.

Lemma related_step (v v' : value) :
  related v v' -> value_step false r statics statics' related v v'.
Proof. exact (value_related_step _ _ _ _ _ _ _ _ memories_valid). Qed.

(* Two related values are the same constant, or neither is one. *)
Lemma related_const_of (v v' : value) :
  related v v' -> const_of v = const_of v'.
Proof.
  intros Hrelated. apply related_step in Hrelated.
  destruct Hrelated
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements];
    try reflexivity; try discriminate.
  all: destruct v; try discriminate; destruct v'; try discriminate;
    reflexivity.
Qed.

Lemma related_const_left (c : const) (v' : value) :
  related (Value_const c) v' -> v' = Value_const c.
Proof.
  intros Hrelated. apply related_const_of in Hrelated.
  destruct v'; try discriminate. injection Hrelated as <-. reflexivity.
Qed.

(* Two related values are locations that correspond, or neither is
   one. *)
Lemma related_location_of (v v' : value) :
  related v v' ->
  match location_of v, location_of v' with
  | Some l, Some l' => corr_locations r l l'
  | None, None => True
  | _, _ => False
  end.
Proof.
  intros Hrelated. apply related_step in Hrelated.
  destruct Hrelated
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements];
    simpl; try exact I; try discriminate.
  - exact Hl.
  - destruct v; try discriminate; destruct v'; try discriminate; exact I.
  - destruct v; try discriminate; destruct v'; try discriminate; exact I.
  - destruct v; try discriminate; destruct v'; try discriminate; exact I.
  - destruct v; try discriminate; destruct v'; try discriminate; exact I.
  - destruct v; try discriminate; destruct v'; try discriminate; exact I.
Qed.

Lemma related_kind (v v' : value) :
  related v v' -> kind_of_value v = kind_of_value v'.
Proof.
  intros Hrelated.
  exact (value_step_kind _ _ _ _ _ _ _ (related_step _ _ Hrelated)).
Qed.

(* What the first of two related values is observed to be, the second
   is. *)
Lemma related_block_of (v v' : value) (tag : Z) (fields : list value) :
  related v v' ->
  block_of statics v = Some (tag, fields) ->
  ∃ fields' : list value,
    block_of statics' v' = Some (tag, fields')
    ∧ Forall2 related fields fields'.
Proof.
  intros Hrelated Hblock. apply related_step in Hrelated.
  destruct Hrelated
    as [c | l l' Hl | v v' tag0 fields0 fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements];
    try discriminate;
    try observations_conflict.
  rewrite Hv in Hblock. injection Hblock as <- <-. exists fields'. auto.
Qed.

Lemma related_boxed_number_of (v v' : value) (c : const) :
  related v v' ->
  boxed_number_of statics v = Some c ->
  boxed_number_of statics' v' = Some c.
Proof.
  intros Hrelated Hboxed. apply related_step in Hrelated.
  destruct Hrelated
    as [c0 | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c0 c' Hv Hv' Hc | c0 | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements];
    try discriminate;
    try observations_conflict.
  - rewrite Hv in Hboxed. injection Hboxed as <-.
    destruct Hc as [<- | [Hfalse _]]; [ exact Hv' | discriminate Hfalse ].
  - exact Hboxed.
Qed.

Lemma related_string_of (v v' : value) (bytes : list Z) :
  related v v' ->
  string_of statics v = Some bytes ->
  string_of statics' v' = Some bytes.
Proof.
  intros Hrelated Hstring. apply related_step in Hrelated.
  destruct Hrelated
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes0 Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements];
    try discriminate;
    try observations_conflict.
  rewrite Hv in Hstring. injection Hstring as <-. exact Hv'.
Qed.

Lemma related_array_of (v v' : value) (elements : list value) :
  related v v' ->
  array_of statics v = Some elements ->
  ∃ elements' : list value,
    array_of statics' v' = Some elements'
    ∧ Forall2 related elements elements'.
Proof.
  intros Hrelated Harray. apply related_step in Hrelated.
  destruct Hrelated
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements0 elements' Hv Hv' Helements];
    try discriminate;
    try observations_conflict.
  rewrite Hv in Harray. injection Harray as <-. exists elements'. auto.
Qed.

Lemma related_closure_of (v v' : value) (code : code_id)
    (slots : list (value_slot * value)) :
  related v v' ->
  closure_of statics v = Some (code, slots) ->
  ∃ slots' : list (value_slot * value),
    closure_of statics' v' = Some (code, slots')
    ∧ map fst slots = map fst slots'
    ∧ Forall2 related (map snd slots) (map snd slots').
Proof.
  intros Hrelated Hclosure. apply related_step in Hrelated.
  destruct Hrelated
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots0 slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements];
    try discriminate;
    try observations_conflict.
  - destruct (closure_in_set_closure_of _ _ _ _ _ Hv)
      as (code0 & Hcode0 & Hclosure0).
    destruct (closure_in_set_closure_of _ _ _ _ _ Hv')
      as (code1 & Hcode1 & Hclosure1).
    rewrite Hclosure0 in Hclosure. injection Hclosure as <- <-.
    rewrite Hcode0 in Hcode1. injection Hcode1 as <-.
    exists slots'. auto.
  - simpl in Hclosure. rewrite Hgslot in Hclosure. discriminate.
Qed.

(* The same of a closure as one of a set. *)
Lemma related_closure_in_set (v v' : value) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  related v v' ->
  closure_in_set statics v = Some (slot, decls, slots) ->
  ∃ slots' : list (value_slot * value),
    closure_in_set statics' v' = Some (slot, decls, slots')
    ∧ map fst slots = map fst slots'
    ∧ Forall2 related (map snd slots) (map snd slots')
    ∧ (∀ (move_to : function_slot) (w : value),
        project_function_slot statics slot move_to v = Some w ->
        ∃ w' : value,
          project_function_slot statics' slot move_to v' = Some w'
          ∧ related w w').
Proof.
  intros Hrelated Hclosure. apply related_step in Hrelated.
  assert (Hprojections :
            ∀ (move_to : function_slot) (w : value),
              project_function_slot statics slot move_to v = Some w ->
              ∃ w' : value,
                project_function_slot statics' slot move_to v' = Some w'
                ∧ related w w').
  { intros move_to w.
    apply (value_step_project false r statics statics' related v v');
      [ intros a a' Hstep; apply Related_step; exact Hstep
      | exact Hrelated ]. }
  destruct Hrelated
    as [c | l l' Hl | v v' tag fields fields' Hv Hv' Hfields
       | v v' c c' Hv Hv' Hc | c | v v' bytes Hv Hv'
       | v v' fslot fdecls slots0 slots' Hv Hv' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hv Hv' Helements];
    try discriminate;
    try observations_conflict.
  - rewrite Hv in Hclosure. injection Hclosure as <- <- <-.
    exists slots'. auto.
  - simpl in Hclosure. rewrite Hgslot in Hclosure. discriminate.
Qed.

(* A value related to a partial application is one, of a related callee
   to related arguments at the same arity. *)
Lemma related_partial_application_of (v v' callee : value)
    (args : list value) (args_arity : arity) :
  related v v' ->
  partial_application_of v = Some (callee, args, args_arity) ->
  ∃ (callee' : value) (args' : list value),
    v' = Value_partial_application callee' args' args_arity
    ∧ related callee callee'
    ∧ Forall2 related args args'.
Proof.
  intros Hrelated Hpartial. apply related_step in Hrelated.
  destruct v as [ | | | | | | | | callee0 args0 arity0];
    simpl in Hpartial; try discriminate Hpartial.
  injection Hpartial as -> -> ->.
  inversion Hrelated; subst; simpl in *; try discriminate.
  eexists. eexists. split; [ reflexivity | split; assumption ].
Qed.

(* And a value related to one that is no partial application is none. *)
Lemma related_not_partial_application (v v' : value) :
  related v v' ->
  partial_application_of v = None ->
  partial_application_of v' = None.
Proof.
  intros Hrelated. apply related_step in Hrelated.
  destruct Hrelated; simpl; intros Hnone; try reflexivity; try discriminate;
    match goal with
    | |- partial_application_of ?w = None =>
      destruct w; simpl in *; try reflexivity; discriminate
    end.
Qed.

Lemma related_is_allocated (v v' : value) :
  related v v' ->
  is_allocated statics v = true ->
  is_allocated statics' v' = true.
Proof.
  intros Hrelated. unfold is_allocated.
  destruct (closure_of statics v) as [[code slots] | ] eqn:Hclosure.
  { destruct (related_closure_of _ _ _ _ Hrelated Hclosure)
      as (slots' & -> & _).
    reflexivity. }
  destruct (block_of statics v) as [[tag fields] | ] eqn:Hblock.
  { destruct (related_block_of _ _ _ _ Hrelated Hblock) as (fields' & -> & _).
    intros _. simpl. rewrite orb_true_r. reflexivity. }
  destruct (boxed_number_of statics v) as [c | ] eqn:Hboxed.
  { rewrite (related_boxed_number_of _ _ _ Hrelated Hboxed).
    intros _. simpl. rewrite !orb_true_r. reflexivity. }
  destruct (string_of statics v) as [bytes | ] eqn:Hstring.
  { rewrite (related_string_of _ _ _ Hrelated Hstring).
    intros _. simpl. rewrite !orb_true_r. reflexivity. }
  destruct (array_of statics v) as [elements | ] eqn:Harray.
  { destruct (related_array_of _ _ _ Hrelated Harray) as (elements' & -> & _).
    intros _. simpl. rewrite !orb_true_r. reflexivity. }
  destruct (partial_application_of v) as [[[callee args] args_arity] | ]
    eqn:Hpartial; [ | discriminate ].
  destruct (related_partial_application_of _ _ _ _ _ Hrelated Hpartial)
    as (callee' & args' & -> & _).
  intros _. reflexivity.
Qed.

Lemma related_int_of_value (kind : standard_int) (v v' : value) :
  related v v' -> int_of_value kind v = int_of_value kind v'.
Proof.
  intros Hrelated.
  rewrite (int_of_value_const_of kind v), (int_of_value_const_of kind v'),
    (related_const_of _ _ Hrelated).
  reflexivity.
Qed.

Lemma related_is_tagged_immediate (v v' : value) :
  related v v' -> is_tagged_immediate v = is_tagged_immediate v'.
Proof.
  intros Hrelated. unfold is_tagged_immediate.
  rewrite (related_int_of_value _ _ _ Hrelated). reflexivity.
Qed.

Lemma related_all_values (vs vs' : list value) :
  Forall2 related vs vs' ->
  forallb (fun v => kind_eqb (kind_of_value v) Value) vs
  = forallb (fun v => kind_eqb (kind_of_value v) Value) vs'.
Proof.
  intros Hall. induction Hall as [ | v v' vs vs' Hv Hall IH]; simpl;
    [ reflexivity | ].
  rewrite (related_kind _ _ Hv), IH. reflexivity.
Qed.

(* Arguments that are bound in the first memory are related to
   arguments that are bound in the second. *)
Lemma related_arguments_bound (vs vs' : list value) :
  Forall2 related vs vs' ->
  forallb (argument_bound statics h) vs = true ->
  forallb (argument_bound statics' h') vs' = true.
Proof.
  intros Hall. induction Hall as [ | v v' vs vs' Hv Hall IH]; simpl;
    [ reflexivity | ].
  rewrite !andb_true_iff. intros [Hbound Hrest].
  split; [ | exact (IH Hrest) ].
  pose proof (related_step _ _ Hv) as Hstep.
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hb Hb' Hfields
       | v v' c c' Hb Hb' Hc | c | v v' bytes Hb Hb'
       | v v' fslot fdecls slots slots' Hb Hb' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hb Hb' Helements]; try reflexivity.
  - destruct (proj1 (proj2 memories_valid) _ _ Hl) as (c & c' & _ & Hc' & _).
    simpl. unfold is_bound. rewrite Hc'. reflexivity.
  - destruct v'; try discriminate; try reflexivity. simpl in Hb' |- *.
    unfold is_bound. destruct (lookup s (static_values statics'));
      [ reflexivity | discriminate ].
  - destruct v'; try discriminate; try reflexivity. simpl in Hb' |- *.
    unfold is_bound. destruct (lookup s (static_values statics'));
      [ reflexivity | discriminate ].
  - destruct v'; try discriminate; try reflexivity. simpl in Hb' |- *.
    unfold is_bound. destruct (lookup s (static_values statics'));
      [ reflexivity | discriminate ].
  - destruct v'; try discriminate; try reflexivity. simpl in Hb' |- *.
    unfold is_bound. destruct (lookup s (static_values statics'));
      [ reflexivity | discriminate ].
  - simpl in Hbound |- *. exact (memories_bound s Hbound).
  - discriminate Hpoison.
  - destruct v'; try discriminate; try reflexivity. simpl in Hb' |- *.
    unfold is_bound. destruct (lookup s (static_values statics'));
      [ reflexivity | discriminate ].
Qed.

(* The same of the test of FlambdaPrimitiveSpec.v: a symbol that is
   bound in the first memory is related to a value that is not a symbol
   bound to nothing in the second. *)
Lemma related_bound_if_symbol (v v' : value) :
  related v v' ->
  bound_if_symbol statics v = true -> bound_if_symbol statics' v' = true.
Proof.
  intros Hv Hbound.
  pose proof (related_step _ _ Hv) as Hstep.
  destruct Hstep
    as [c | l l' Hl | v v' tag fields fields' Hb Hb' Hfields
       | v v' c c' Hb Hb' Hc | c | v v' bytes Hb Hb'
       | v v' fslot fdecls slots slots' Hb Hb' Hnames Hslots
         Hsiblings
       | gslot gdecls gslots Hgslot | s Hs Hs'
       | kind v' Hpoison Hkind
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | v v' elements elements' Hb Hb' Helements]; try reflexivity.
  1-4:
    destruct v'; try discriminate; try reflexivity; simpl in Hb' |- *;
    destruct (lookup s (static_values statics'));
    [ reflexivity | discriminate ].
  - simpl in Hbound |- *. pose proof (memories_bound s) as Hdomain.
    unfold is_bound in Hdomain.
    destruct (lookup s (static_values statics));
      [ | discriminate Hbound ].
    destruct (lookup s (static_values statics'));
      [ reflexivity | discriminate (Hdomain eq_refl) ].
  - discriminate Hpoison.
  - destruct v'; try discriminate; try reflexivity; simpl in Hb' |- *;
      destruct (lookup s (static_values statics'));
      [ reflexivity | discriminate ].
Qed.

Lemma related_all_bound_if_symbol (vs vs' : list value) :
  Forall2 related vs vs' ->
  forallb (bound_if_symbol statics) vs = true ->
  forallb (bound_if_symbol statics') vs' = true.
Proof.
  intros Hall. induction Hall as [ | v v' vs vs' Hv Hall IH]; simpl;
    [ reflexivity | ].
  rewrite !andb_true_iff. intros [Hbound Hrest].
  split; [ exact (related_bound_if_symbol _ _ Hv Hbound) | exact (IH Hrest) ].
Qed.

(* The primitives that do not touch the heap, on related arguments. *)
Lemma related_load_field (kind : block_access_kind) (field : Z) (tag : Z)
    (fields fields' : list value) (loaded : value) :
  Forall2 related fields fields' ->
  load_field kind field tag fields = Some loaded ->
  ∃ loaded' : value,
    load_field kind field tag fields' = Some loaded' ∧ related loaded loaded'.
Proof.
  intros Hfields. destruct kind as [known_tag size field_kind]. simpl.
  rewrite <- (Forall2_length Hfields).
  destruct (_ && _); [ | discriminate ].
  destruct (nth_error fields (Z.to_nat field)) as [v | ] eqn:Hnth; simpl;
    [ | discriminate ].
  destruct (Forall2_nth_error _ _ _ _ _ Hfields Hnth) as (v' & -> & Hv).
  simpl.
  assert (Hcheck :
            match field_kind with
            | Any_value => kind_eqb (kind_of_value v') Value
            | Immediate => is_tagged_immediate v'
            end
            = match field_kind with
              | Any_value => kind_eqb (kind_of_value v) Value
              | Immediate => is_tagged_immediate v
              end).
  { destruct field_kind;
      [ rewrite (related_kind _ _ Hv)
      | rewrite (related_is_tagged_immediate _ _ Hv) ];
      reflexivity. }
  rewrite Hcheck.
  destruct
    match field_kind with
    | Any_value => kind_eqb (kind_of_value v) Value
    | Immediate => is_tagged_immediate v
    end;
    [ | discriminate ].
  intros [= <-]. exists v'. auto.
Qed.

Lemma related_store_field (kind : block_access_kind) (field : Z)
    (new new' : value) (tag : Z) (fields fields' fields1 : list value) :
  related new new' ->
  Forall2 related fields fields' ->
  store_field kind field new tag fields = Some fields1 ->
  ∃ fields1' : list value,
    store_field kind field new' tag fields' = Some fields1'
    ∧ Forall2 related fields1 fields1'.
Proof.
  intros Hnew Hfields. destruct kind as [known_tag size field_kind]. simpl.
  rewrite <- (Forall2_length Hfields).
  assert (Hcheck :
            match field_kind with
            | Any_value => kind_eqb (kind_of_value new') Value
            | Immediate => is_tagged_immediate new'
            end
            = match field_kind with
              | Any_value => kind_eqb (kind_of_value new) Value
              | Immediate => is_tagged_immediate new
              end).
  { destruct field_kind;
      [ rewrite (related_kind _ _ Hnew)
      | rewrite (related_is_tagged_immediate _ _ Hnew) ];
      reflexivity. }
  rewrite Hcheck.
  destruct (_ && _); [ | discriminate ].
  apply FlambdaPrimitiveProofs.Forall2_replace_nth; assumption.
Qed.

(* The relation is one that the observations respect
   (FlambdaPrimitiveSpec.respects_observations), so every primitive that
   does not touch the heap computes related values of related
   arguments: FlambdaPrimitiveSpec.eval_prim_related_statement. *)
Lemma related_respects_observations :
  respects_observations related statics statics'.
Proof.
  split; [ split | split; [ split; [ | split ] | ] ].
  - intros c v' Hv. exact (related_const_left c v' Hv).
  - intros v c Hv. pose proof (related_const_of _ _ Hv) as Hconst.
    destruct v; try discriminate. simpl in Hconst. injection Hconst as ->.
    reflexivity.
  - intros c.
    split; [ apply related_const | apply Related_step, Step_same_boxed_number ].
  - intros tag fields fields' Hfields.
    apply (related_block _ _ _ _ _ _ tag fields fields');
      [ reflexivity | reflexivity | exact Hfields ].
  - intros elements elements' Helements. apply Related_step.
    exact
      (Step_array _ _ _ _ _ (Value_immutable_array elements)
         (Value_immutable_array elements') elements elements' eq_refl
         eq_refl Helements).
  - intros v v' Hv. split; [ | split; [ | split; [ | split ] ] ].
    + intros tag fields Hblock. exact (related_block_of _ _ _ _ Hv Hblock).
    + intros c Hboxed. exact (related_boxed_number_of _ _ _ Hv Hboxed).
    + intros bytes Hstring. exact (related_string_of _ _ _ Hv Hstring).
    + intros Hclosure.
      destruct (closure_of statics v) as [[code slots] | ] eqn:Hc;
        [ | contradiction Hclosure; reflexivity ].
      destruct (related_closure_of _ _ _ _ Hv Hc) as (slots' & -> & _).
      discriminate.
    + split; [ | split; [ exact (related_bound_if_symbol _ _ Hv) | split ] ].
      * intros Hpartial.
        destruct (partial_application_of v) as [[[callee args] args_arity] | ]
          eqn:Hp; [ | contradiction Hpartial; reflexivity ].
        destruct (related_partial_application_of _ _ _ _ _ Hv Hp)
          as (callee' & args' & -> & _).
        discriminate.
      * intros [s c] Hs.
        apply FlambdaPrimitiveProofs.imported_of_symbol in Hs as [-> Hs].
        pose proof (related_step _ _ Hv) as Hstep.
        assert (Hnot : is_allocated statics (Value_symbol s) = false).
        { unfold is_allocated. simpl. rewrite Hs. reflexivity. }
        inversion Hstep; subst;
          try (unfold is_allocated in Hnot; simpl in Hnot;
               match goal with
               | H : _ = Some _ |- _ =>
                 simpl in H; rewrite Hs in H; discriminate H
               end).
        simpl. rewrite (proj1 (memories_imported s c) Hs). reflexivity.
      * intros elements Harray. exact (related_array_of _ _ _ Hv Harray).
Qed.

(* So it keeps the kind and the integers, which is what the functions of
   arrays ask of a relation (FlambdaPrimitiveSpec.relates_numbers). *)
Lemma related_numbers : relates_numbers related.
Proof.
  apply FlambdaPrimitiveProofs.relates_constants_numbers.
  exact (proj1 related_respects_observations).
Qed.

Lemma related_imported_of (v v' : value) (i : symbol * import_class) :
  related v v' ->
  imported_of statics v = Some i ->
  imported_of statics' v' = Some i.
Proof.
  intros Hv.
  exact
    (proj1
       (proj2
          (proj2
             (proj2
                (proj2
                   (proj2
                      (proj2
                         (proj2 (proj2 related_respects_observations) v v'
                            Hv)))))))
       i).
Qed.

Lemma related_eval_prim (p : prim) (vs vs' : list value) (result : value) :
  Forall2 related vs vs' ->
  eval_prim statics p vs = Some result ->
  ∃ result' : value,
    eval_prim statics' p vs' = Some result' ∧ related result result'.
Proof.
  exact
    (FlambdaPrimitiveProofs.eval_prim_related related statics statics' p vs
       vs' result related_respects_observations).
Qed.

(* [Phys_equal] with a location among its arguments. *)
Lemma related_is_location (v v' : value) :
  related v v' -> is_location v' = is_location v.
Proof.
  intros Hv. pose proof (related_location_of _ _ Hv) as Hl.
  destruct v, v'; simpl in Hl |- *; try reflexivity; contradiction.
Qed.

Lemma related_phys_equal_on_location (p : binary_primitive)
    (vs vs' : list value) :
  Forall2 related vs vs' ->
  phys_equal_on_location p vs' = phys_equal_on_location p vs.
Proof.
  intros Hvs.
  destruct p; try reflexivity.
  destruct Hvs as [ | v1 v1' vs vs' Hv1 Hvs]; [ reflexivity | ].
  destruct Hvs as [ | v2 v2' vs vs' Hv2 Hvs]; [ reflexivity | ].
  destruct Hvs as [ | v3 v3' vs vs' Hv3 Hvs]; [ | reflexivity ].
  simpl. rewrite (related_is_location _ _ Hv1), (related_is_location _ _ Hv2).
  reflexivity.
Qed.

Lemma related_eval_phys_equal_location (v1 v1' v2 v2' : value) (b : bool) :
  related v1 v1' ->
  related v2 v2' ->
  eval_phys_equal_location statics v1 v2 = Some b ->
  eval_phys_equal_location statics' v1' v2' = Some b.
Proof.
  intros Hfirst Hsecond. rewrite !eval_phys_equal_location_cases.
  pose proof (related_location_of _ _ Hfirst) as Hl1.
  pose proof (related_location_of _ _ Hsecond) as Hl2.
  rewrite <- (related_int_of_value Standard_tagged_immediate _ _ Hfirst),
    <- (related_int_of_value Standard_tagged_immediate _ _ Hsecond).
  destruct (location_of v1) as [l1 | ], (location_of v1') as [l1' | ];
    try contradiction;
    destruct (location_of v2) as [l2 | ], (location_of v2') as [l2' | ];
    try contradiction.
  - intros [= <-]. f_equal.
    pose proof (proj1 memories_valid _ _ _ _ Hl1 Hl2) as Hiff.
    destruct (Pos.eqb_spec l1 l2) as [Heq | Hneq],
      (Pos.eqb_spec l1' l2') as [Heq' | Hneq']; try reflexivity.
    + exfalso. apply Hneq'. apply Hiff. exact Heq.
    + exfalso. apply Hneq. apply Hiff. exact Heq'.
  - destruct (is_some (int_of_value Standard_tagged_immediate v2)); simpl;
      [ auto | ].
    destruct (is_allocated statics v2) eqn:Hallocated;
      [ rewrite (related_is_allocated _ _ Hsecond Hallocated); auto | ].
    unfold is_imported_pointer.
    destruct (imported_of statics v2) as [[s c] | ] eqn:Hi;
      [ | discriminate ].
    rewrite (related_imported_of _ _ _ Hsecond Hi).
    destruct c; simpl; try discriminate; rewrite orb_true_r; auto.
  - destruct (is_some (int_of_value Standard_tagged_immediate v1)); simpl;
      [ auto | ].
    destruct (is_allocated statics v1) eqn:Hallocated;
      [ rewrite (related_is_allocated _ _ Hfirst Hallocated); auto | ].
    unfold is_imported_pointer.
    destruct (imported_of statics v1) as [[s c] | ] eqn:Hi;
      [ | discriminate ].
    rewrite (related_imported_of _ _ _ Hfirst Hi).
    destruct c; simpl; try discriminate; rewrite orb_true_r; auto.
  - discriminate.
Qed.

Lemma related_exec_unary_on_cell (p : unary_primitive) (l l' : location)
    (c c' : cell) (result : value) :
  corr_locations r l l' ->
  cell_related false r statics statics' c c' ->
  exec_unary_on_cell p l c = Some result ->
  ∃ result' : value,
    exec_unary_on_cell p l' c' = Some result' ∧ related result result'.
Proof.
  intros Hl Hcell.
  destruct c as [cell_mut tag fields | akind elements],
    c' as [cell_mut' tag' fields' | akind' elements'];
    try (exfalso; exact Hcell).
  - destruct Hcell as (<- & <- & Hfields).
    destruct p; simpl; try discriminate.
    + destruct (load_allowed _ cell_mut); [ | discriminate ].
      apply related_load_field. exact Hfields.
    + intros [= <-]. eexists. split; [ reflexivity | apply related_const ].
    + intros [= <-]. eexists. split; [ reflexivity | apply related_const ].
    + intros [= <-]. eexists. split; [ reflexivity | apply related_const ].
    + destruct (kind_eqb _ Value); [ | discriminate ].
      intros [= <-]. eexists.
      split; [ reflexivity | apply related_location; exact Hl ].
  - (* An array: [Is_int] without [variant_only], [Is_null], the length
       and the identity. *)
    destruct Hcell as (<- & Helements).
    destruct p as [ | [ | ] | | | length_kind | | okind | | | | | | | | ];
      simpl; try discriminate.
    + intros [= <-]. eexists. split; [ reflexivity | apply related_const ].
    + intros [= <-]. eexists. split; [ reflexivity | apply related_const ].
    + rewrite
        (FlambdaPrimitiveProofs.array_length_of_same_length elements
           elements' (eq_sym (Forall2_length Helements))).
      unfold array_length_of.
      destruct (in_range _ _); [ | discriminate ].
      intros [= <-]. eexists. split; [ reflexivity | apply related_const ].
    + destruct (kind_eqb _ Value); [ | discriminate ].
      intros [= <-]. eexists.
      split; [ reflexivity | apply related_location; exact Hl ].
Qed.

(* A binary primitive that reads a cell of the first memory reads one of
   the second, at a location that corresponds and with a related
   argument; and one that reads no cell of the first reads none of the
   second. *)
Lemma related_binary_on_cell (p : binary_primitive) (vs vs' : list value) :
  Forall2 related vs vs' ->
  match binary_on_cell p vs, binary_on_cell p vs' with
  | Some (l, v2), Some (l', v2') => corr_locations r l l' ∧ related v2 v2'
  | None, None => True
  | _, _ => False
  end.
Proof.
  intros Hvs.
  destruct Hvs as [ | v1 v1' vs vs' Hv1 Hvs]; [ destruct p; exact I | ].
  destruct Hvs as [ | v2 v2' vs vs' Hv2 Hvs];
    [ destruct p, v1, v1'; exact I | ].
  destruct Hvs as [ | v3 v3' vs vs' Hv3 Hvs];
    [ | destruct p, v1, v1'; exact I ].
  pose proof (related_location_of _ _ Hv1) as Hl.
  destruct p, v1, v1'; simpl in Hl |- *; try exact I; try contradiction;
    split; assumption.
Qed.

Lemma related_exec_binary_on_cell (p : binary_primitive) (l l' : location)
    (c c' : cell) (v2 v2' result : value) :
  corr_locations r l l' ->
  cell_related false r statics statics' c c' ->
  related v2 v2' ->
  exec_binary_on_cell p l c v2 = Some result ->
  ∃ result' : value,
    exec_binary_on_cell p l' c' v2' = Some result' ∧ related result result'.
Proof.
  intros Hl Hcell Hv2.
  destruct c as [cell_mut tag fields | akind elements],
    c' as [cell_mut' tag' fields' | akind' elements'];
    try (exfalso; exact Hcell); [ destruct p; simpl; discriminate | ].
  destruct Hcell as (<- & Helements).
  destruct p as [array_kind load_kind mut | | | | | ]; simpl;
    try discriminate.
  destruct (load_allowed mut Cell_mutable); [ | discriminate ].
  exact
    (FlambdaPrimitiveProofs.array_loaded_related related load_kind elements
       elements' v2 v2' result related_numbers Helements Hv2).
Qed.

(* What a primitive leaves: memories related under a correspondence
   that extends the one before, and related results. *)
Definition exec_related (r1 : correspondence) (h1 h1' : heap)
    (result result' : value) : Prop :=
  memories_related D r1 statics statics' h1 h1'
  ∧ (∀ l l' : location, corr_locations r l l' -> corr_locations r1 l l')
  ∧ corr_values r1 = corr_values r
  ∧ value_related false r1 statics statics' result result'.

Lemma exec_related_same (result result' : value) :
  related result result' -> exec_related r h h' result result'.
Proof.
  intros Hresult. split; [ exact Hmemories | ]. split; [ auto | ].
  split; [ reflexivity | exact Hresult ].
Qed.

(* The correspondence with one more pair of locations. *)
Definition add_location (l l' : location) : correspondence :=
  Correspondence
    (fun a a' => corr_locations r a a' ∨ (a = l ∧ a' = l'))
    (corr_values r).

Lemma exec_related_allocation (l l' : location) (c c' : cell) :
  lookup l h = None ->
  lookup l' h' = None ->
  cell_related false r statics statics' c c' ->
  exec_related (add_location l l') ((l, c) :: h) ((l', c') :: h')
    (Value_location l) (Value_location l').
Proof.
  intros Hl Hl' Hnew.
  assert (Hgrow :
            ∀ a a' : value,
              related a a' ->
              value_related false (add_location l l') statics statics' a a').
  { intros a a'.
    apply value_related_grow; [ intros b b' Hb; left; exact Hb | auto ]. }
  assert (Hgrow_cell :
            ∀ a a' : cell,
              cell_related false r statics statics' a a' ->
              cell_related false (add_location l l') statics statics' a a').
  { intros [m t fs | k es] [m' t' fs' | k' es']; simpl;
      try exact (fun H => H).
    - intros (Hm & Ht & Hfs). split; [ exact Hm | ]. split; [ exact Ht | ].
      revert Hfs. apply Forall2_mono. exact Hgrow.
    - intros (Hk & Hes). split; [ exact Hk | ].
      revert Hes. apply Forall2_mono. exact Hgrow. }
  pose proof memories_valid as (Hbijection & Hcells & Hvalues).
  split; [ split; [ split; [ | split ] | ] | split; [ | split ] ].
  - intros a a2 b b2 [Hab | [-> ->]] [Hab2 | [-> ->]].
    + exact (Hbijection _ _ _ _ Hab Hab2).
    + destruct (Hcells _ _ Hab) as (c0 & c0' & Hc & Hc' & _).
      split; intros ->; exfalso.
      * rewrite Hl in Hc. discriminate.
      * rewrite Hl' in Hc'. discriminate.
    + destruct (Hcells _ _ Hab2) as (c0 & c0' & Hc & Hc' & _).
      split; intros <-; exfalso.
      * rewrite Hl in Hc. discriminate.
      * rewrite Hl' in Hc'. discriminate.
    + split; intros _; reflexivity.
  - intros a a' [Hab | [-> ->]].
    + destruct (Hcells _ _ Hab) as (c0 & c0' & Hc & Hc' & Hcell).
      exists c0, c0'. simpl.
      destruct (Pos.eqb_spec a l) as [-> | _];
        [ rewrite Hl in Hc; discriminate | ].
      destruct (Pos.eqb_spec a' l') as [-> | _];
        [ rewrite Hl' in Hc'; discriminate | ].
      split; [ exact Hc | ]. split; [ exact Hc' | ].
      exact (Hgrow_cell _ _ Hcell).
    + exists c, c'. simpl.
      rewrite !Pos.eqb_refl. split; [ reflexivity | ].
      split; [ reflexivity | ]. exact (Hgrow_cell _ _ Hnew).
  - intros a a' Ha. generalize (Hvalues _ _ Ha).
    apply value_step_grow; [ intros b b' Hb; left; exact Hb | exact Hgrow ].
  - exact memories_names.
  - intros a a' Ha. left. exact Ha.
  - reflexivity.
  - apply related_location. right. split; reflexivity.
Qed.

Lemma exec_related_write (l l' : location) (c c' : cell) :
  corr_locations r l l' ->
  cell_related false r statics statics' c c' ->
  exec_related r ((l, c) :: h) ((l', c') :: h') unit_value unit_value.
Proof.
  intros Hl Hcell.
  pose proof memories_valid as (Hbijection & Hcells & Hvalues).
  split;
    [ split;
      [ split; [ exact Hbijection | split; [ | exact Hvalues ] ]
      | exact memories_names ]
    | split; [ auto | split; [ reflexivity | apply related_const ] ] ].
  intros a a' Hab. simpl.
  pose proof (Hbijection _ _ _ _ Hab Hl) as Hiff.
  destruct (Pos.eqb_spec a l) as [Heq | Hneq],
    (Pos.eqb_spec a' l') as [Heq' | Hneq'].
  - eexists. eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
    exact Hcell.
  - exfalso. apply Hneq'. apply Hiff. exact Heq.
  - exfalso. apply Hneq. apply Hiff. exact Heq'.
  - exact (Hcells _ _ Hab).
Qed.

Lemma related_exec_pure (p : prim) (vs vs' : list value) (result : value)
    (h1 : heap) :
  Forall2 related vs vs' ->
  (let* v := eval_prim statics p vs in Some (v, h)) = Some (result, h1) ->
  ∃ (result' : value) (h1' : heap) (r1 : correspondence),
    (let* v := eval_prim statics' p vs' in Some (v, h')) = Some (result', h1')
    ∧ exec_related r1 h1 h1' result result'.
Proof.
  intros Hvs.
  destruct (eval_prim statics p vs) as [v | ] eqn:Heval; simpl;
    [ | discriminate ].
  intros [= <- <-].
  destruct (related_eval_prim _ _ _ _ Hvs Heval) as (v' & -> & Hv).
  exists v', h', r. split; [ reflexivity | apply exec_related_same; exact Hv ].
Qed.

(* Every primitive respects the relation. *)
Lemma related_exec_prim (p : prim) (vs vs' : list value) (result : value)
    (h1 : heap) :
  Forall2 related vs vs' ->
  exec_prim statics h p vs = Some (result, h1) ->
  ∃ (result' : value) (h1' : heap) (r1 : correspondence),
    exec_prim statics' h' p vs' = Some (result', h1')
    ∧ exec_related r1 h1 h1' result result'.
Proof.
  intros Hvs Hexec. apply exec_prim_bound in Hexec as [Hbound Hexec].
  unfold exec_prim. rewrite (related_arguments_bound _ _ Hvs Hbound).
  clear Hbound. revert Hexec.
  destruct p as [p | p | p | p | rp].
  - destruct Hvs as [ | v v' vs vs' Hv Hvs];
      [ exact (related_exec_pure (Unary p) [] [] result h1 (Forall2_nil _))
      | ].
    destruct Hvs as [ | v2 v2' vs vs' Hv2 Hvs].
    2: { destruct v; simpl; discriminate. }
    rewrite !exec_bound_prim_unary.
    pose proof (related_location_of _ _ Hv) as Hl.
    destruct (location_of v) as [l | ], (location_of v') as [l' | ];
      try contradiction.
    + destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
      destruct (proj1 (proj2 memories_valid) _ _ Hl)
        as (c0 & c' & Hc0 & Hc' & Hcell).
      rewrite Hc in Hc0. injection Hc0 as <-. rewrite Hc'. simpl.
      destruct (exec_unary_on_cell p l c) as [res | ] eqn:Hres; simpl;
        [ | discriminate ].
      intros [= <- <-].
      destruct (related_exec_unary_on_cell _ _ _ _ _ _ Hl Hcell Hres)
        as (res' & -> & Hres').
      exists res', h', r.
      split; [ reflexivity | apply exec_related_same; exact Hres' ].
    + exact
        (related_exec_pure (Unary p) [v] [v'] result h1
           (Forall2_cons _ _ Hv (Forall2_nil _))).
  - rewrite !exec_bound_prim_binary, (related_phys_equal_on_location p _ _ Hvs).
    destruct (phys_equal_on_location p vs).
    + destruct p; try discriminate.
      destruct Hvs as [ | v1 v1' vs vs' Hv1 Hvs]; [ discriminate | ].
      destruct Hvs as [ | v2 v2' vs vs' Hv2 Hvs]; [ discriminate | ].
      destruct Hvs as [ | v3 v3' vs vs' Hv3 Hvs]; [ | discriminate ].
      destruct (eval_phys_equal_location statics v1 v2) as [b | ] eqn:Hb;
        simpl; [ | discriminate ].
      rewrite (related_eval_phys_equal_location _ _ _ _ _ Hv1 Hv2 Hb). simpl.
      intros [= <- <-]. eexists. exists h', r.
      split; [ reflexivity | apply exec_related_same, related_const ].
    + (* A binary primitive that reads a cell, or one that reads none. *)
      pose proof (related_binary_on_cell p _ _ Hvs) as Hon.
      destruct (binary_on_cell p vs) as [[l v2] | ],
        (binary_on_cell p vs') as [[l' v2'] | ]; try contradiction;
        [ | exact (related_exec_pure (Binary p) vs vs' result h1 Hvs) ].
      destruct Hon as [Hl Hv2].
      destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
      destruct (proj1 (proj2 memories_valid) _ _ Hl)
        as (c0 & c' & Hc0 & Hc' & Hcell).
      rewrite Hc in Hc0. injection Hc0 as <-. rewrite Hc'. simpl.
      destruct (exec_binary_on_cell p l c v2) as [res | ] eqn:Hres; simpl;
        [ | discriminate ].
      intros [= <- <-].
      destruct
        (related_exec_binary_on_cell _ _ _ _ _ _ _ _ Hl Hcell Hv2 Hres)
        as (res' & -> & Hres').
      exists res', h', r.
      split; [ reflexivity | apply exec_related_same; exact Hres' ].
  - exact (related_exec_pure (Variadic p) vs vs' result h1 Hvs).
  - change (exec_bound_prim statics h (Heap p) vs)
      with (exec_heap_primitive statics h p vs).
    change (exec_bound_prim statics' h' (Heap p) vs')
      with (exec_heap_primitive statics' h' p vs').
    destruct p as [mut tag | kind init field | kind | akind set_kind].
    + simpl. rewrite <- (related_all_values _ _ Hvs).
      destruct (forallb _ vs); [ | discriminate ]. intros [= <- <-].
      eexists. eexists.
      exists
        (add_location (FlambdaHeapSpec.fresh_location h)
           (FlambdaHeapSpec.fresh_location h')).
      split; [ reflexivity | ].
      apply exec_related_allocation;
        [ exact (fresh_location h) | exact (fresh_location h')
        | simpl; auto ].
    + destruct Hvs as [ | v v' vs vs' Hv Hvs]; [ simpl; discriminate | ].
      destruct Hvs as [ | new new' vs vs' Hnew Hvs]; [ simpl; discriminate | ].
      destruct Hvs as [ | v3 v3' vs vs' Hv3 Hvs]; [ | simpl; discriminate ].
      rewrite !exec_block_set.
      pose proof (related_location_of _ _ Hv) as Hl.
      destruct (location_of v) as [l | ], (location_of v') as [l' | ];
        try contradiction; [ | discriminate ].
      destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ]
        eqn:Hc;
        try discriminate.
      destruct (proj1 (proj2 memories_valid) _ _ Hl)
        as (c0 & c' & Hc0 & Hc' & Hcell).
      rewrite Hc in Hc0. injection Hc0 as <-. rewrite Hc'.
      destruct c' as [mut' tag' fields' | akind' elements'];
        [ | exfalso; exact Hcell ].
      destruct Hcell as (<- & <- & Hfields).
      destruct (store_field kind field new tag fields) as [fields1 | ]
        eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct (related_store_field _ _ _ _ _ _ _ _ Hnew Hfields Hstore)
        as (fields1' & -> & Hfields1).
      simpl. eexists. eexists. exists r.
      split;
        [ reflexivity | apply exec_related_write; [ exact Hl | simpl; auto ] ].
    + (* [Make_mutable_array] *)
      simpl.
      destruct Hvs as [ | v0 v0' vs vs' Hv0 Hvs]; [ discriminate | ].
      destruct (array_made kind (v0 :: vs)) as [elements | ] eqn:Hmade;
        simpl; [ | discriminate ].
      intros [= <- <-].
      destruct
        (FlambdaPrimitiveProofs.array_made_related related kind (v0 :: vs)
           (v0' :: vs') elements related_numbers (Forall2_cons _ _ Hv0 Hvs)
           Hmade)
        as (elements' & -> & Helements).
      simpl. eexists. eexists.
      exists
        (add_location (FlambdaHeapSpec.fresh_location h)
           (FlambdaHeapSpec.fresh_location h')).
      split; [ reflexivity | ].
      apply exec_related_allocation;
        [ exact (fresh_location h) | exact (fresh_location h')
        | simpl; auto ].
    + (* [Array_set] *)
      destruct Hvs as [ | v v' vs vs' Hv Hvs]; [ simpl; discriminate | ].
      destruct Hvs as [ | index index' vs vs' Hindex Hvs];
        [ simpl; discriminate | ].
      destruct Hvs as [ | new new' vs vs' Hnew Hvs]; [ simpl; discriminate | ].
      destruct Hvs as [ | v4 v4' vs vs' Hv4 Hvs]; [ | simpl; discriminate ].
      rewrite !exec_array_set.
      pose proof (related_location_of _ _ Hv) as Hl.
      destruct (location_of v) as [l | ], (location_of v') as [l' | ];
        try contradiction; [ | discriminate ].
      destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:Hc;
        try discriminate.
      destruct (proj1 (proj2 memories_valid) _ _ Hl)
        as (c0 & c' & Hc0 & Hc' & Hcell).
      rewrite Hc in Hc0. injection Hc0 as <-. rewrite Hc'.
      destruct c' as [mut' tag' fields' | kind' elements'];
        [ exfalso; exact Hcell | ].
      destruct Hcell as (<- & Helements).
      destruct (array_stored set_kind kind elements index new)
        as [stored | ] eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct
        (FlambdaPrimitiveProofs.array_stored_related related set_kind kind
           elements elements' index index' new new' stored related_numbers
           Helements Hindex Hnew Hstore)
        as (stored' & -> & Hstored).
      simpl. eexists. eexists. exists r.
      split;
        [ reflexivity | apply exec_related_write; [ exact Hl | simpl; auto ] ].
  - intros Hexec. discriminate Hexec.
Qed.

(* Evaluation under related environments. *)
Lemma related_eval_simple (needed : variable -> bool) (vars vars' : env)
    (s : simple) (v : value) :
  env_related related needed vars vars' ->
  (∀ x : variable, var_in_simple x s = true -> needed x = true) ->
  simple_avoids D s = true ->
  eval_simple vars s = Some v ->
  ∃ v' : value, eval_simple vars' s = Some v' ∧ related v v'.
Proof.
  intros Henv Hneeded Havoids. destruct s as [x | sym | c]; simpl.
  - intros Hx. apply (Henv x v); [ | exact Hx ].
    apply Hneeded. simpl. apply Pos.eqb_refl.
  - intros [= <-]. eexists.
    split; [ reflexivity | apply Related_corr, memories_symbols ].
    exact Havoids.
  - intros [= <-]. eexists. split; [ reflexivity | apply related_const ].
Qed.

Lemma related_eval_simples (needed : variable -> bool) (vars vars' : env)
    (args : list simple) (vs : list value) :
  env_related related needed vars vars' ->
  (∀ x : variable, var_in_simples x args = true -> needed x = true) ->
  simples_avoid D args = true ->
  eval_simples vars args = Some vs ->
  ∃ vs' : list value,
    eval_simples vars' args = Some vs' ∧ Forall2 related vs vs'.
Proof.
  intros Henv. revert vs.
  induction args as [ | s args IH]; intros vs Hneeded Havoid; simpl.
  - intros [= <-]. exists []. split; [ reflexivity | constructor ].
  - apply andb_true_iff in Havoid as [Havoid_head Havoid_tail].
    destruct (eval_simple vars s) as [v | ] eqn:Hv; simpl; [ | discriminate ].
    destruct (eval_simples vars args) as [vs0 | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-].
    assert (Hhead :
              ∀ x : variable, var_in_simple x s = true -> needed x = true).
    { intros x Hx. apply Hneeded. unfold var_in_simples. simpl.
      unfold var_in_simple in Hx. rewrite Hx. reflexivity. }
    assert (Htail :
              ∀ x : variable, var_in_simples x args = true -> needed x = true).
    { intros x Hx. apply Hneeded. unfold var_in_simples in Hx |- *. simpl.
      rewrite Hx. apply orb_true_r. }
    destruct (related_eval_simple _ _ _ _ _ Henv Hhead Havoid_head Hv)
      as (v' & -> & Hv').
    destruct (IH _ Htail Havoid_tail eq_refl) as (vs' & -> & Hvs').
    simpl. eexists. split; [ reflexivity | constructor; assumption ].
Qed.

Lemma related_eval_value_slots (needed : variable -> bool) (vars vars' : env)
    (slots : list (value_slot * simple)) (vs : list (value_slot * value)) :
  env_related related needed vars vars' ->
  (∀ x : variable, var_in_value_slots x slots = true -> needed x = true) ->
  value_slots_avoid D slots = true ->
  eval_value_slots vars slots = Some vs ->
  ∃ vs' : list (value_slot * value),
    eval_value_slots vars' slots = Some vs'
    ∧ map fst vs = map fst vs'
    ∧ Forall2 related (map snd vs) (map snd vs').
Proof.
  intros Henv. revert vs.
  induction slots as [ | [slot s] slots IH]; intros vs Hneeded Havoid; simpl.
  - intros [= <-]. exists [].
    split; [ reflexivity | split; [ reflexivity | constructor ] ].
  - apply andb_true_iff in Havoid as [Havoid_head Havoid_tail].
    simpl in Havoid_head.
    destruct (eval_simple vars s) as [v | ] eqn:Hv; simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs0 | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-].
    assert (Hhead :
              ∀ x : variable, var_in_simple x s = true -> needed x = true).
    { intros x Hx. apply Hneeded. unfold var_in_value_slots. simpl.
      rewrite Hx. reflexivity. }
    assert (Htail :
              ∀ x : variable,
                var_in_value_slots x slots = true -> needed x = true).
    { intros x Hx. apply Hneeded. unfold var_in_value_slots in Hx |- *.
      simpl. rewrite Hx. apply orb_true_r. }
    destruct (related_eval_simple _ _ _ _ _ Henv Hhead Havoid_head Hv)
      as (v' & -> & Hv').
    destruct (IH _ Htail Havoid_tail eq_refl) as (vs' & -> & Hnames & Hvs').
    simpl. eexists. split; [ reflexivity | ]. simpl.
    split; [ rewrite Hnames; reflexivity | constructor; assumption ].
Qed.

(* A named expression respects the relation. *)
Lemma related_heap_exec_named (needed : variable -> bool) (vars vars' : env)
    (d : named) (result : value) (h1 : heap) :
  env_related related needed vars vars' ->
  (∀ x : variable, var_in_named x d = true -> needed x = true) ->
  named_avoids D d = true ->
  heap_exec_named statics h vars d = Some (result, h1) ->
  ∃ (result' : value) (h1' : heap) (r1 : correspondence),
    heap_exec_named statics' h' vars' d = Some (result', h1')
    ∧ exec_related r1 h1 h1' result result'.
Proof.
  intros Henv Hneeded Havoid.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | project_from slot closure | move_from move_to closure];
    simpl in Hneeded, Havoid |- *.
  - destruct (eval_simple vars s) as [v | ] eqn:Hv; simpl; [ | discriminate ].
    intros [= <- <-].
    destruct (related_eval_simple _ _ _ _ _ Henv Hneeded Havoid Hv)
      as (v' & -> & Hv').
    simpl. exists v', h', r.
    split; [ reflexivity | apply exec_related_same; exact Hv' ].
  - destruct (eval_simples vars args) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    destruct (related_eval_simples _ _ _ _ _ Henv Hneeded Havoid Hvs)
      as (vs' & -> & Hvs').
    simpl. apply related_exec_prim. exact Hvs'.
  - apply andb_true_iff in Havoid as [_ Havoid].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <- <-].
    destruct (related_eval_value_slots _ _ _ _ _ Henv Hneeded Havoid Hvs)
      as (vs' & -> & Hnames & Hvs').
    simpl. eexists. exists h', r.
    split; [ reflexivity | apply exec_related_same ].
    apply
      (related_closure _ _ _ _ _ _ unnamed_function_slot
         [(unnamed_function_slot, code)] vs vs');
      [ reflexivity | reflexivity | exact Hnames | exact Hvs'
      | left; split; exact I ].
  - apply andb_true_iff in Havoid as [_ Havoid].
    destruct (lookup fslot decls) as [code | ] eqn:Hfslot; simpl;
      [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <- <-].
    destruct (related_eval_value_slots _ _ _ _ _ Henv Hneeded Havoid Hvs)
      as (vs' & -> & Hnames & Hvs').
    simpl. eexists. exists h', r.
    split; [ reflexivity | apply exec_related_same ].
    assert (Hclosure :
              ∀ (st : static_env) (ws : list (value_slot * value)),
                closure_in_set st (Value_closure_of_set fslot decls ws)
                = Some (fslot, decls, ws)).
    { intros st ws. simpl. rewrite Hfslot. reflexivity. }
    apply (related_closure _ _ _ _ _ _ fslot decls vs vs');
      [ apply Hclosure | apply Hclosure | exact Hnames | exact Hvs'
      | left; split; exact I ].
  - destruct (eval_simple vars closure) as [v | ] eqn:Hv; simpl;
      [ | discriminate ].
    destruct (related_eval_simple _ _ _ _ _ Henv Hneeded Havoid Hv)
      as (v' & -> & Hv').
    simpl.
    destruct (closure_of statics v) as [[code slots] | ] eqn:Hclosure; simpl;
      [ | discriminate ].
    destruct (related_closure_of _ _ _ _ Hv' Hclosure)
      as (slots' & -> & Hnames & Hslots).
    simpl.
    destruct (lookup slot slots) as [w | ] eqn:Hw; simpl; [ | discriminate ].
    intros [= <- <-].
    destruct (lookup_related _ _ _ _ _ Hnames Hslots Hw) as (w' & -> & Hw').
    exists w', h', r.
    split; [ reflexivity | apply exec_related_same; exact Hw' ].
  - destruct (eval_simple vars closure) as [v | ] eqn:Hv; simpl;
      [ | discriminate ].
    destruct (related_eval_simple _ _ _ _ _ Henv Hneeded Havoid Hv)
      as (v' & -> & Hv').
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics v) as [[[at_slot decls] slots] | ]
      eqn:Hclosure; simpl; [ | discriminate ].
    destruct (related_closure_in_set _ _ _ _ _ Hv' Hclosure)
      as (slots' & -> & Hnames & Hslots & _).
    simpl. destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    destruct (lookup slot slots) as [w | ] eqn:Hw; simpl; [ | discriminate ].
    intros [= <- <-].
    destruct (lookup_related _ _ _ _ _ Hnames Hslots Hw) as (w' & -> & Hw').
    exists w', h', r.
    split; [ reflexivity | apply exec_related_same; exact Hw' ].
  - destruct (eval_simple vars closure) as [v | ] eqn:Hv; simpl;
      [ | discriminate ].
    destruct (related_eval_simple _ _ _ _ _ Henv Hneeded Havoid Hv)
      as (v' & -> & Hv').
    simpl.
    destruct (project_function_slot statics move_from move_to v) as [w | ]
      eqn:Hw; simpl; [ | discriminate ].
    intros [= <- <-].
    destruct (project_function_slot_closure _ _ _ _ _ Hw)
      as (decls & slots & Hclosure).
    destruct (related_closure_in_set _ _ _ _ _ Hv' Hclosure)
      as (slots' & Hclosure' & Hnames & Hslots & Hsiblings).
    destruct (Hsiblings _ _ Hw) as (w' & Hw' & Hrelated).
    rewrite Hw'. simpl. exists w', h', r.
    split; [ reflexivity | apply exec_related_same; exact Hrelated ].
Qed.

(* What two symbols bound by the same static binding are bound to. *)
Definition static_related (R : value -> value -> Prop)
    (a a' : statically_allocated_value) : Prop :=
  match a, a' with
  | Static_closure code slots, Static_closure code' slots' =>
    code = code'
    ∧ map fst slots = map fst slots'
    ∧ Forall2 R (map snd slots) (map snd slots')
  | Static_closure_of_set slot syms decls slots,
    Static_closure_of_set slot' syms' decls' slots' =>
    slot = slot' ∧ syms = syms' ∧ decls = decls'
    ∧ map fst slots = map fst slots'
    ∧ Forall2 R (map snd slots) (map snd slots')
  | Static_block tag fields, Static_block tag' fields' =>
    tag = tag' ∧ Forall2 R fields fields'
  | Static_boxed_number v, Static_boxed_number v' => R v v'
  | Static_string bytes, Static_string bytes' => bytes = bytes'
  | Static_immutable_array elements, Static_immutable_array elements' =>
    Forall2 R elements elements'
  | _, _ => False
  end.

Lemma related_eval_static_const (needed : variable -> bool) (vars vars' : env)
    (s : symbol) (const : static_const) (a : statically_allocated_value) :
  env_related related needed vars vars' ->
  (∀ x : variable,
    var_in_static_binding x (Bind_symbol s const) = true ->
    needed x = true) ->
  static_const_avoids D const = true ->
  eval_static_const vars const = Some a ->
  ∃ a' : statically_allocated_value,
    eval_static_const vars' const = Some a' ∧ static_related related a a'.
Proof.
  intros Henv Hneeded Havoid.
  destruct const
    as [code slots | slot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl in Hneeded, Havoid |- *.
  - apply andb_true_iff in Havoid as [_ Havoid].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-].
    destruct (related_eval_value_slots _ _ _ _ _ Henv Hneeded Havoid Hvs)
      as (vs' & -> & Hnames & Hvs').
    simpl. eexists. split; [ reflexivity | ]. simpl. auto.
  - apply andb_true_iff in Havoid as [_ Havoid].
    destruct (lookup slot decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-].
    destruct (related_eval_value_slots _ _ _ _ _ Henv Hneeded Havoid Hvs)
      as (vs' & -> & Hnames & Hvs').
    simpl. eexists. split; [ reflexivity | ]. simpl. auto 6.
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-].
    destruct (related_eval_simples _ _ _ _ _ Henv Hneeded Havoid Hvs)
      as (vs' & -> & Hvs').
    simpl. eexists. split; [ reflexivity | ]. simpl. auto.
  - destruct (eval_simple vars contents) as [v | ] eqn:Hv; simpl;
      [ | discriminate ].
    intros [= <-].
    destruct (related_eval_simple _ _ _ _ _ Henv Hneeded Havoid Hv)
      as (v' & -> & Hv').
    simpl. eexists. split; [ reflexivity | exact Hv' ].
  - intros [= <-]. eexists. split; reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-].
    destruct (related_eval_simples _ _ _ _ _ Henv Hneeded Havoid Hvs)
      as (vs' & -> & Hvs').
    simpl. eexists. split; [ reflexivity | exact Hvs' ].
Qed.

(* The symbols that a static set of closures names are those that its
   constant names. *)
Lemma eval_static_const_sets_kept (vars : env) (const : static_const)
    (slot : function_slot) (syms : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  static_const_avoids D const = true ->
  eval_static_const vars const
  = Some (Static_closure_of_set slot syms decls slots) ->
  closure_symbols_avoid D syms = true.
Proof.
  intros Havoid.
  destruct const
    as [code cslots | cslot csyms cdecls cslots | tag fields | contents
       | bytes | elements];
    simpl in Havoid |- *.
  - destruct (eval_value_slots vars cslots); discriminate.
  - destruct (lookup cslot cdecls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars cslots); simpl; [ | discriminate ].
    intros [= _ <- _ _].
    apply andb_true_iff in Havoid as [Havoid _].
    apply andb_true_iff in Havoid as [Havoid _]. exact Havoid.
  - destruct (eval_simples vars fields); discriminate.
  - destruct (eval_simple vars contents); discriminate.
  - discriminate.
  - destruct (eval_simples vars elements); discriminate.
Qed.

(* The static environment with one more symbol. *)
Definition bind_symbol (st : static_env) (s : symbol)
    (a : statically_allocated_value) : static_env :=
  Static_env (static_code st) ((s, a) :: static_values st).

(* Two memories stay related when a symbol that neither binds comes to
   be bound, in each, to related things. *)
Lemma memories_bind_symbol (s : symbol) (a a' : statically_allocated_value) :
  symbol_kept D s = true ->
  (∀ (slot : function_slot) (syms : list (function_slot * symbol))
     (decls : list (function_slot * code_id))
     (slots : list (value_slot * value)),
    a = Static_closure_of_set slot syms decls slots ->
    closure_symbols_avoid D syms = true) ->
  lookup s (static_values statics) = None ->
  static_related related a a' ->
  memories_related D r (bind_symbol statics s a) (bind_symbol statics' s a')
    h h'.
Proof.
  intros Hkept Hsets Hs Ha.
  assert (Hs' : lookup s (static_values statics') = None).
  { pose proof (memories_domain s Hkept) as Hdomain.
    unfold is_bound in Hdomain.
    rewrite Hs in Hdomain.
    destruct (lookup s (static_values statics'));
      [ discriminate | reflexivity ]. }
  assert (Hextend : statics_extend statics (bind_symbol statics s a)).
  { intros s0 a0 Hlookup. simpl.
    destruct (Pos.eqb_spec s0 s) as [-> | _];
      [ rewrite Hs in Hlookup; discriminate | exact Hlookup ]. }
  assert (Hextend' : statics_extend statics' (bind_symbol statics' s a')).
  { intros s0 a0 Hlookup. simpl.
    destruct (Pos.eqb_spec s0 s) as [-> | _];
      [ rewrite Hs' in Hlookup; discriminate | exact Hlookup ]. }
  assert (Hmono :
            ∀ v v' : value,
              related v v' ->
              value_related false r (bind_symbol statics s a)
                (bind_symbol statics' s a') v v').
  { intros v v'.
    apply value_related_statics;
      [ exact Hextend | exact Hextend' | exact memories_opaque ]. }
  assert (Hnew :
            value_step false r (bind_symbol statics s a)
              (bind_symbol statics' s a')
              (value_related false r (bind_symbol statics s a)
                 (bind_symbol statics' s a'))
              (Value_symbol s) (Value_symbol s)).
  { destruct a
      as [code slots | slot syms decls slots | tag fields | contents
         | bytes | elements | ],
      a'
      as [code' slots' | slot' syms' decls' slots' | tag' fields'
         | contents' | bytes' | elements' | ];
      simpl in Ha; try contradiction.
    - destruct Ha as (<- & Hnames & Hslots).
      apply
        (Step_closure _ _ _ _ _ _ _ unnamed_function_slot
           [(unnamed_function_slot, code)] slots slots');
        [ simpl; rewrite Pos.eqb_refl; reflexivity
        | simpl; rewrite Pos.eqb_refl; reflexivity
        | exact Hnames
        | | ].
      + revert Hslots. apply Forall2_mono. exact Hmono.
      + left. split; unfold set_bound; simpl; rewrite Pos.eqb_refl; exact I.
    - destruct Ha as (<- & <- & <- & Hnames & Hslots).
      destruct (lookup slot decls) as [code | ] eqn:Hslot.
      + apply (Step_closure _ _ _ _ _ _ _ slot decls slots slots');
          [ cbn -[Pos.eqb]; rewrite Pos.eqb_refl, Hslot; reflexivity
          | cbn -[Pos.eqb]; rewrite Pos.eqb_refl, Hslot; reflexivity
          | exact Hnames
          | | ].
        * revert Hslots. apply Forall2_mono. exact Hmono.
        * right. split; intros move_to; cbn -[Pos.eqb];
            rewrite !Pos.eqb_refl, Hslot; cbn -[Pos.eqb];
            [ | intros Hnone; exact Hnone ].
          intros w.
          destruct (lookup move_to decls); cbn -[Pos.eqb]; [ | discriminate ].
          destruct (lookup move_to syms) as [t | ] eqn:Ht; cbn -[Pos.eqb];
            [ intros [= <-] | discriminate ].
          eexists. split; [ reflexivity | ].
          apply Related_corr, memories_symbols.
          pose proof (Hsets _ _ _ _ eq_refl) as Hsyms.
          unfold closure_symbols_avoid in Hsyms.
          rewrite forallb_forall in Hsyms.
          exact (Hsyms (move_to, t) (lookup_in _ _ _ Ht)).
      + apply Step_opaque_symbol; unfold is_allocated; cbn -[Pos.eqb];
          rewrite Pos.eqb_refl, Hslot; reflexivity.
    - destruct Ha as (<- & Hfields).
      apply (Step_block _ _ _ _ _ _ _ tag fields fields');
        [ simpl; rewrite Pos.eqb_refl; reflexivity
        | simpl; rewrite Pos.eqb_refl; reflexivity
        | ].
      revert Hfields. apply Forall2_mono. exact Hmono.
    - pose proof (related_const_of _ _ Ha) as Hconst.
      destruct (const_of contents) as [c | ] eqn:Hc.
      + destruct contents; try discriminate. injection Hc as ->.
        destruct contents'; try discriminate. injection Hconst as <-.
        destruct (boxable_contents c) as [c1 | ] eqn:Hboxable.
        * apply (Step_boxed_number _ _ _ _ _ _ _ c1 c1);
            [ simpl; rewrite Pos.eqb_refl; exact Hboxable
            | simpl; rewrite Pos.eqb_refl; exact Hboxable
            | left; reflexivity ].
        * apply Step_opaque_symbol; unfold is_allocated; simpl;
            rewrite Pos.eqb_refl, Hboxable; reflexivity.
      + apply Step_opaque_symbol; unfold is_allocated; simpl;
          rewrite Pos.eqb_refl.
        * destruct contents; try discriminate; reflexivity.
        * destruct contents'; try discriminate; reflexivity.
    - subst bytes'.
      apply (Step_string _ _ _ _ _ _ _ bytes); simpl; rewrite Pos.eqb_refl;
        reflexivity.
    - apply (Step_array _ _ _ _ _ _ _ elements elements');
        [ simpl; rewrite Pos.eqb_refl; reflexivity
        | simpl; rewrite Pos.eqb_refl; reflexivity
        | ].
      revert Ha. apply Forall2_mono. exact Hmono. }
  pose proof memories_valid as (Hbijection & Hcells & Hvalues).
  split;
    [ split; [ exact Hbijection | split ]
    | split; [ | split; [ | split ] ] ].
  - intros l l' Hl. destruct (Hcells _ _ Hl) as (c & c' & Hc & Hc' & Hcell).
    exists c, c'. split; [ exact Hc | ]. split; [ exact Hc' | ].
    exact (cell_related_change _ _ _ _ _ _ _ _ _ _ Hmono Hcell).
  - intros v v' Hv. pose proof (Hvalues _ _ Hv) as Hstep.
    destruct Hstep
      as [c | l l' Hl | v v' tag fields fields' Hb Hb' Hfields
         | v v' c c' Hb Hb' Hc | c | v v' bytes Hb Hb'
         | v v' fslot fdecls slots slots' Hb Hb' Hnames Hslots
           Hsiblings
         | gslot gdecls gslots Hgslot | s0 Hs0 Hs0'
         | kind v' Hpoison Hkind
         | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
         | v v' elements elements' Hb Hb' Helements].
    + apply Step_const.
    + apply Step_location. exact Hl.
    + exact
        (Step_block _ _ _ _ _ _ _ tag fields fields'
           (block_of_extend _ _ _ _ Hextend Hb)
           (block_of_extend _ _ _ _ Hextend' Hb')
           (Forall2_mono _ _ _ _ Hmono Hfields)).
    + exact
        (Step_boxed_number _ _ _ _ _ _ _ c c'
           (boxed_number_of_extend _ _ _ _ Hextend Hb)
           (boxed_number_of_extend _ _ _ _ Hextend' Hb') Hc).
    + apply Step_same_boxed_number.
    + exact
        (Step_string _ _ _ _ _ _ _ bytes
           (string_of_extend _ _ _ _ Hextend Hb)
           (string_of_extend _ _ _ _ Hextend' Hb')).
    + apply
        (Step_closure _ _ _ _ _ _ _ fslot fdecls slots slots'
           (closure_in_set_extend _ _ _ _ Hextend Hb)
           (closure_in_set_extend _ _ _ _ Hextend' Hb') Hnames
           (Forall2_mono _ _ _ _ Hmono Hslots)).
      destruct Hsiblings as [[Hbound Hbound'] | [Hsiblings Hnone]].
      * left.
        split;
          [ exact (set_bound_extend _ _ _ _ Hextend Hb Hbound)
          | exact (set_bound_extend _ _ _ _ Hextend' Hb' Hbound') ].
      * right. split; intros move_to;
          rewrite (project_function_slot_extend _ _ _ _ _ _ Hextend Hb),
            (project_function_slot_extend _ _ _ _ _ _ Hextend' Hb');
          [ | apply Hnone ].
        intros w Hw. destruct (Hsiblings _ _ Hw) as (w' & Hw' & Hrelated).
        exists w'. split; [ exact Hw' | exact (Hmono _ _ Hrelated) ].
    + apply Step_same_closure_of_set. exact Hgslot.
    + destruct (Pos.eqb_spec s0 s) as [-> | Hneq]; [ exact Hnew | ].
      apply Step_opaque_symbol.
      * unfold is_allocated in Hs0 |- *. simpl in Hs0 |- *.
        destruct (Pos.eqb_spec s0 s) as [Heq | _];
          [ contradiction | exact Hs0 ].
      * unfold is_allocated in Hs0' |- *. simpl in Hs0' |- *.
        destruct (Pos.eqb_spec s0 s) as [Heq | _];
          [ contradiction | exact Hs0' ].
    + discriminate Hpoison.
    + apply Step_partial_application;
        [ exact (Hmono _ _ Hpcallee)
        | exact (Forall2_mono _ _ _ _ Hmono Hpargs) ].
    + exact
        (Step_array _ _ _ _ _ _ _ elements elements'
           (array_of_extend _ _ _ _ Hextend Hb)
           (array_of_extend _ _ _ _ Hextend' Hb')
           (Forall2_mono _ _ _ _ Hmono Helements)).
  - exact memories_symbols.
  - intros s0 Hkept0. unfold is_bound. simpl.
    destruct (Pos.eqb s0 s);
      [ reflexivity | exact (memories_domain s0 Hkept0) ].
  - intros s0 Hdropped.
    destruct (memories_left_out s0 Hdropped) as [Hnone Hallocated].
    assert (Hneq : s0 <> s).
    { intros ->. rewrite Hkept in Hdropped. discriminate. }
    split.
    + simpl. destruct (Pos.eqb_spec s0 s) as [Heq | _];
        [ contradiction | exact Hnone ].
    + unfold is_allocated in Hallocated |- *. simpl in Hallocated |- *.
      destruct (Pos.eqb_spec s0 s) as [Heq | _];
        [ contradiction | exact Hallocated ].
  - intros s1 c. cbn -[Pos.eqb].
    destruct (Pos.eqb s1 s); [ | exact (memories_imported s1 c) ].
    split; intros [= ->]; [ | destruct a ]; simpl in Ha; contradiction.
Qed.

End Machine.

(* The relations on environments and continuations are monotone. *)
Lemma env_related_mono (R R1 : value -> value -> Prop)
    (needed needed1 : variable -> bool) (vars vars' : env) :
  (∀ v v' : value, R v v' -> R1 v v') ->
  (∀ x : variable, needed1 x = true -> needed x = true) ->
  env_related R needed vars vars' ->
  env_related R1 needed1 vars vars'.
Proof.
  intros Hmono Hneeded Henv x v Hx Hv.
  destruct (Henv x v (Hneeded x Hx) Hv) as (v' & Hv' & Hrelated).
  exists v'. auto.
Qed.

(* Induction on a continuation, through the continuations it holds. *)
Lemma cont_value_induction (P : cont_value -> Prop) :
  (∀ (handler : continuation_handler) (vars : env)
     (conts : list (continuation * cont_value))
     (recursive : list (continuation * continuation_handler)),
    Forall (fun b => P (snd b)) conts ->
    P (Closure handler vars conts recursive)) ->
  (∀ k : continuation, P (Outer k)) ->
  (∀ (c : cont_value) (extra : list value), P c -> P (Extra_args c extra)) ->
  (∀ (remaining : list value) (remaining_arity return_arity : arity)
     (return_to exn_to : cont_value),
    P return_to ->
    P exn_to ->
    P (Over_application remaining remaining_arity return_arity return_to
         exn_to)) ->
  ∀ c : cont_value, P c.
Proof.
  intros Hclosure Houter Hextra Hover. fix IH 1.
  intros [handler vars conts recursive | k | c extra
         | remaining remaining_arity return_arity return_to exn_to].
  - apply Hclosure.
    exact
      ((fix go (conts : list (continuation * cont_value))
          : Forall (fun b => P (snd b)) conts :=
          match conts with
          | [] => Forall_nil _
          | b :: conts => Forall_cons _ (IH (snd b)) (go conts)
          end)
         conts).
  - apply Houter.
  - apply Hextra. apply IH.
  - apply Hover; apply IH.
Qed.

Lemma Forall2_impl_Forall {A B : Type} (R R1 : A -> B -> Prop) (l : list A)
    (l' : list B) :
  Forall (fun a => ∀ b : B, R a b -> R1 a b) l ->
  Forall2 R l l' ->
  Forall2 R1 l l'.
Proof.
  intros Hall Hrelated.
  induction Hrelated as [ | a b l l' Hab Hrelated IH]; constructor;
    inversion Hall as [ | a0 l0 Ha Hrest]; subst; auto.
Qed.

Lemma cont_related_mono (R R1 : value -> value -> Prop) (c c' : cont_value) :
  (∀ v v' : value, R v v' -> R1 v v') ->
  cont_related R c c' ->
  cont_related R1 c c'.
Proof.
  intros Hmono. revert c'.
  induction c
    as [handler vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_induction; intros c' Hc;
    inversion Hc
      as [k0 | handler0 vars0 vars' conts0 conts' recursive0 Henv Hconts
         | c0 c0' extra0 extra' Hc0 Hextra
         | remaining0 remaining' remaining_arity0 return_arity0 return_to0
             return_to' exn_to0 exn_to' Hremaining Hreturn Hexn];
    subst.
  - apply Related_closure.
    + revert Henv. apply env_related_mono; [ exact Hmono | auto ].
    + revert Hconts. apply Forall2_impl_Forall. revert IH.
      apply Forall_impl. intros b Hb b' [Hname Hrelated].
      split; [ exact Hname | exact (Hb _ Hrelated) ].
  - apply Related_outer.
  - apply Related_extra_args; [ exact (IH _ Hc0) | ].
    revert Hextra. apply Forall2_mono. exact Hmono.
  - apply Related_over_application;
      [ revert Hremaining; apply Forall2_mono; exact Hmono
      | exact (IHreturn _ Hreturn) | exact (IHexn _ Hexn) ].
Qed.

Lemma conts_related_mono (R R1 : value -> value -> Prop)
    (conts conts' : cont_env) :
  (∀ v v' : value, R v v' -> R1 v v') ->
  conts_related R conts conts' ->
  conts_related R1 conts conts'.
Proof.
  intros Hmono. apply Forall2_mono. intros b b' [Hname Hrelated].
  split; [ exact Hname | exact (cont_related_mono _ _ _ _ Hmono Hrelated) ].
Qed.

Lemma related_resolve (R : value -> value -> Prop) (k : continuation)
    (conts conts' : cont_env) :
  conts_related R conts conts' ->
  cont_related R (resolve k conts) (resolve k conts').
Proof.
  intros Hconts. unfold conts_related in Hconts. unfold resolve.
  induction Hconts as [ | [k1 c] [k1' c'] conts conts' [Hname Hc] Hconts IH];
    simpl.
  - apply Related_outer.
  - simpl in Hname. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

Lemma related_with_extra_args (R : value -> value -> Prop) (c c' : cont_value)
    (extra extra' : list value) :
  cont_related R c c' ->
  Forall2 R extra extra' ->
  cont_related R (with_extra_args c extra) (with_extra_args c' extra').
Proof.
  intros Hc Hextra.
  destruct Hextra as [ | v v' extra extra' Hv Hextra]; simpl; [ exact Hc | ].
  apply Related_extra_args; [ exact Hc | constructor; assumption ].
Qed.

Lemma related_rec_closures (R : value -> value -> Prop)
    (recursive : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  env_related R
    (fun x => existsb (fun '(_, h) => var_free_in_handler_b x h) recursive)
    vars vars' ->
  conts_related R conts conts' ->
  conts_related R (rec_closures recursive vars conts)
    (rec_closures recursive vars' conts').
Proof.
  intros Henv Hconts. unfold rec_closures, conts_related.
  assert (Hall :
            ∀ handlers : list (continuation * continuation_handler),
              (∀ b, b ∈ handlers -> b ∈ recursive) ->
              Forall2
                (fun b b' =>
                   fst b = fst b' ∧ cont_related R (snd b) (snd b'))
                (map (fun '(k, h) => (k, Closure h vars conts recursive))
                   handlers)
                (map (fun '(k, h) => (k, Closure h vars' conts' recursive))
                   handlers)).
  { induction handlers as [ | [k handler] handlers IH]; intros Hin; simpl;
      constructor.
    - split; [ reflexivity | ]. simpl.
      apply Related_closure; [ | exact Hconts ].
      intros x v Hx. apply Henv. unfold needed_by_closure in Hx.
      apply orb_true_iff in Hx as [Hx | Hx]; [ | exact Hx ].
      apply existsb_exists. exists (k, handler).
      split; [ apply Hin; left; reflexivity | exact Hx ].
    - apply IH. intros b Hb. apply Hin. right. exact Hb. }
  apply Hall. auto.
Qed.

Lemma related_bind_params (R : value -> value -> Prop)
    (needed : variable -> bool) (params : list variable)
    (vs vs' : list value) (vars vars' env1 : env) :
  Forall2 R vs vs' ->
  env_related R (fun x => needed x && negb (among x params)) vars vars' ->
  bind_params params vs vars = Some env1 ->
  ∃ env1' : env,
    bind_params params vs' vars' = Some env1' ∧ env_related R needed env1 env1'.
Proof.
  intros Hvs. revert params vars vars'.
  induction Hvs as [ | v v' vs vs' Hv Hvs IH]; intros [ | x params] vars vars'
    Henv; simpl; try discriminate.
  - intros [= <-]. exists vars'. split; [ reflexivity | ].
    revert Henv. apply env_related_mono; [ auto | ].
    intros y Hy. simpl. rewrite Hy. reflexivity.
  - apply IH. intros y w Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | Hneq].
    + intros [= <-]. exists v'. auto.
    + apply Henv. unfold among. simpl.
      rewrite (proj2 (Pos.eqb_neq y x) Hneq). exact Hy.
Qed.

Lemma find_arm_in (i : Z) (arms : list (Z * apply_cont)) (ac : apply_cont) :
  find_arm i arms = Some ac -> ∃ d : Z, (d, ac) ∈ arms.
Proof.
  induction arms as [ | [d ac0] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb i d).
  - intros [= <-]. exists d. left. reflexivity.
  - intros Hfind. destruct (IH Hfind) as (d0 & Hd0). exists d0. right.
    exact Hd0.
Qed.

(* Whether a value is allocated depends on the static values only. *)
Lemma is_allocated_same_values (st st2 : static_env) (v : value) :
  static_values st2 = static_values st ->
  is_allocated st2 v = is_allocated st v.
Proof.
  intros Hsame.
  unfold is_allocated, closure_of, block_of, boxed_number_of, string_of,
    array_of.
  rewrite Hsame. reflexivity.
Qed.

(* Two memories stay related when the code of their static environments
   changes. *)
Lemma memories_same_values (D : dropped) (r : correspondence)
    (st st' st2 st2' : static_env) (h h' : heap) :
  static_values st2 = static_values st ->
  static_values st2' = static_values st' ->
  memories_related D r st st' h h' ->
  memories_related D r st2 st2' h h'.
Proof.
  intros Hsame Hsame' Hmemories.
  pose proof Hmemories
    as ((Hbijection & Hcells & Hvalues) & Hsymbols & Hdomain & Hleft_out
        & Himported).
  assert (Hextend : statics_extend st st2).
  { intros s a Hlookup. rewrite Hsame. exact Hlookup. }
  assert (Hextend' : statics_extend st' st2').
  { intros s a Hlookup. rewrite Hsame'. exact Hlookup. }
  assert (Hmono :
            ∀ v v' : value,
              value_related false r st st' v v' ->
              value_related false r st2 st2' v v').
  { intros v v'.
    apply value_related_statics;
      [ exact Hextend | exact Hextend'
      | exact (memories_opaque D r st st' h h' Hmemories) ]. }
  split;
    [ split; [ exact Hbijection | split ]
    | split; [ | split; [ | split ] ] ].
  - intros l l' Hl. destruct (Hcells _ _ Hl) as (c & c' & Hc & Hc' & Hcell).
    exists c, c'. split; [ exact Hc | ]. split; [ exact Hc' | ].
    exact (cell_related_change _ _ _ _ _ _ _ _ _ _ Hmono Hcell).
  - intros v v' Hv. pose proof (Hvalues _ _ Hv) as Hstep.
    destruct Hstep
      as [c | l l' Hl | v v' tag fields fields' Hb Hb' Hfields
         | v v' c c' Hb Hb' Hc | c | v v' bytes Hb Hb'
         | v v' fslot fdecls slots slots' Hb Hb' Hnames Hslots
           Hsiblings
         | gslot gdecls gslots Hgslot | s0 Hs0 Hs0'
         | kind v' Hpoison Hkind
         | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
         | v v' elements elements' Hb Hb' Helements].
    + apply Step_const.
    + apply Step_location. exact Hl.
    + exact
        (Step_block _ _ _ _ _ _ _ tag fields fields'
           (block_of_extend _ _ _ _ Hextend Hb)
           (block_of_extend _ _ _ _ Hextend' Hb')
           (Forall2_mono _ _ _ _ Hmono Hfields)).
    + exact
        (Step_boxed_number _ _ _ _ _ _ _ c c'
           (boxed_number_of_extend _ _ _ _ Hextend Hb)
           (boxed_number_of_extend _ _ _ _ Hextend' Hb') Hc).
    + apply Step_same_boxed_number.
    + exact
        (Step_string _ _ _ _ _ _ _ bytes
           (string_of_extend _ _ _ _ Hextend Hb)
           (string_of_extend _ _ _ _ Hextend' Hb')).
    + apply
        (Step_closure _ _ _ _ _ _ _ fslot fdecls slots slots'
           (closure_in_set_extend _ _ _ _ Hextend Hb)
           (closure_in_set_extend _ _ _ _ Hextend' Hb') Hnames
           (Forall2_mono _ _ _ _ Hmono Hslots)).
      destruct Hsiblings as [[Hbound Hbound'] | [Hsiblings Hnone]].
      * left.
        split;
          [ exact (set_bound_extend _ _ _ _ Hextend Hb Hbound)
          | exact (set_bound_extend _ _ _ _ Hextend' Hb' Hbound') ].
      * right. split; intros move_to;
          rewrite (project_function_slot_extend _ _ _ _ _ _ Hextend Hb),
            (project_function_slot_extend _ _ _ _ _ _ Hextend' Hb');
          [ | apply Hnone ].
        intros w Hw. destruct (Hsiblings _ _ Hw) as (w' & Hw' & Hrelated).
        exists w'. split; [ exact Hw' | exact (Hmono _ _ Hrelated) ].
    + apply Step_same_closure_of_set. exact Hgslot.
    + apply Step_opaque_symbol.
      * rewrite (is_allocated_same_values _ _ _ Hsame). exact Hs0.
      * rewrite (is_allocated_same_values _ _ _ Hsame'). exact Hs0'.
    + discriminate Hpoison.
    + apply Step_partial_application;
        [ exact (Hmono _ _ Hpcallee)
        | exact (Forall2_mono _ _ _ _ Hmono Hpargs) ].
    + exact
        (Step_array _ _ _ _ _ _ _ elements elements'
           (array_of_extend _ _ _ _ Hextend Hb)
           (array_of_extend _ _ _ _ Hextend' Hb')
           (Forall2_mono _ _ _ _ Hmono Helements)).
  - exact Hsymbols.
  - intros s Hkept. rewrite Hsame, Hsame'. exact (Hdomain s Hkept).
  - intros s Hdropped. rewrite Hsame.
    rewrite (is_allocated_same_values _ _ _ Hsame').
    exact (Hleft_out s Hdropped).
  - intros s. rewrite Hsame, Hsame'. exact (Himported s).
Qed.

(* What the bindings of a group name, outside the code they bind: the
   symbols they bind are not left out, and their constants refer to no
   symbol left out. *)
Definition binding_names_kept (D : dropped) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_symbol s const => symbol_kept D s && static_const_avoids D const
  | Bind_code _ _ | Bind_deleted_code _ => true
  end.

(* A group of static bindings, run under related environments, leaves
   related memories. *)
Lemma related_eval_static_bindings (D : dropped) (r : correspondence)
    (h h' : heap) (needed : variable -> bool) (vars vars' : env)
    (bindings : list static_binding) :
  ∀ st st' st1 : static_env,
    memories_related D r st st' h h' ->
    static_code st = static_code st' ->
    env_related (value_related false r st st') needed vars vars' ->
    (∀ x : variable,
      existsb (var_in_static_binding x) bindings = true ->
      needed x = true) ->
    forallb (binding_names_kept D) bindings = true ->
    eval_static_bindings vars bindings st = Some st1 ->
    ∃ st1' : static_env,
      eval_static_bindings vars' bindings st' = Some st1'
      ∧ memories_related D r st1 st1' h h'
      ∧ static_code st1 = static_code st1'
      ∧ statics_extend st st1
      ∧ statics_extend st' st1'.
Proof.
  induction bindings as [ | binding bindings IH];
    intros st st' st1 Hmemories Hcode Henv Hneeded Hkept; simpl.
  - intros [= <-]. exists st'. split; [ reflexivity | ].
    split; [ exact Hmemories | ]. split; [ exact Hcode | ].
    split; intros s a Hlookup; exact Hlookup.
  - simpl in Hkept. apply andb_true_iff in Hkept as [Hkept_head Hkept_tail].
    assert (Htail :
              ∀ x : variable,
                existsb (var_in_static_binding x) bindings = true ->
                needed x = true).
    { intros x Hx. apply Hneeded.
      change
        (var_in_static_binding x binding
         || existsb (var_in_static_binding x) bindings
         = true).
      rewrite Hx. apply orb_true_r. }
    destruct binding as [id c | id | s const].
    + rewrite <- Hcode.
      destruct (lookup id (static_code st)); [ discriminate | ].
      intros Heval.
      pose (st2 := Static_env ((id, c) :: static_code st) (static_values st)).
      pose
        (st2' := Static_env ((id, c) :: static_code st) (static_values st')).
      assert (Hmemories2 : memories_related D r st2 st2' h h').
      { apply (memories_same_values D r st st');
          [ reflexivity | reflexivity | exact Hmemories ]. }
      assert (Henv2 :
                env_related (value_related false r st2 st2') needed vars vars').
      { revert Henv. apply env_related_mono; [ | auto ]. intros v v'.
        apply value_related_statics;
          [ intros s a Hlookup; exact Hlookup
          | intros s a Hlookup; exact Hlookup
          | exact (memories_opaque D r st st' h h' Hmemories) ]. }
      destruct
        (IH st2 st2' st1 Hmemories2 eq_refl Henv2 Htail Hkept_tail Heval)
        as (st1' & Heval' & Hmemories1 & Hcode1 & Hextend & Hextend').
      exists st1'. split; [ exact Heval' | ]. split; [ exact Hmemories1 | ].
      split; [ exact Hcode1 | ]. split; [ exact Hextend | exact Hextend' ].
    + exact (IH st st' st1 Hmemories Hcode Henv Htail Hkept_tail).
    + simpl in Hkept_head.
      apply andb_true_iff in Hkept_head as [Hkept_s Havoid_const].
      destruct (lookup s (static_values st)) as [a0 | ] eqn:Hs;
        [ discriminate | ].
      assert (Hs' : lookup s (static_values st') = None).
      { pose proof (memories_domain D r st st' h h' Hmemories s Hkept_s)
          as Hdomain.
        unfold is_bound in Hdomain. rewrite Hs in Hdomain.
        destruct (lookup s (static_values st'));
          [ discriminate | reflexivity ]. }
      rewrite Hs'.
      destruct (eval_static_const vars const) as [a | ] eqn:Ha; simpl;
        [ | discriminate ].
      assert (Hhead :
                ∀ x : variable,
                  var_in_static_binding x (Bind_symbol s const) = true ->
                  needed x = true).
      { intros x Hx. apply Hneeded.
        change
          (var_in_static_binding x (Bind_symbol s const)
           || existsb (var_in_static_binding x) bindings
           = true).
        rewrite Hx. reflexivity. }
      destruct
        (related_eval_static_const D r st st' h h' Hmemories needed vars
           vars' s const a Henv Hhead Havoid_const Ha)
        as (a' & -> & Hrelated).
      simpl. intros Heval.
      assert (Hsets :
                ∀ (slot : function_slot) (syms : list (function_slot * symbol))
                  (decls : list (function_slot * code_id))
                  (slots : list (value_slot * value)),
                  a = Static_closure_of_set slot syms decls slots ->
                  closure_symbols_avoid D syms = true).
      { intros slot syms decls slots ->.
        exact
          (eval_static_const_sets_kept D vars const _ _ _ _ Havoid_const
             Ha). }
      pose proof (memories_bind_symbol D r st st' h h' Hmemories s a a'
                    Hkept_s Hsets Hs Hrelated)
        as Hmemories2.
      assert (Hextend2 : statics_extend st (bind_symbol st s a)).
      { intros s0 a0 Hlookup. simpl.
        destruct (Pos.eqb_spec s0 s) as [-> | _];
          [ rewrite Hs in Hlookup; discriminate | exact Hlookup ]. }
      assert (Hextend2' : statics_extend st' (bind_symbol st' s a')).
      { intros s0 a0 Hlookup. simpl.
        destruct (Pos.eqb_spec s0 s) as [-> | _];
          [ rewrite Hs' in Hlookup; discriminate | exact Hlookup ]. }
      assert (Henv2 :
                env_related
                  (value_related false r (bind_symbol st s a)
                     (bind_symbol st' s a'))
                  needed vars vars').
      { revert Henv. apply env_related_mono; [ | auto ]. intros v v'.
        apply value_related_statics;
          [ exact Hextend2
          | exact Hextend2'
          | exact (memories_opaque D r st st' h h' Hmemories) ]. }
      destruct
        (IH (bind_symbol st s a) (bind_symbol st' s a') st1 Hmemories2 Hcode
           Henv2 Htail Hkept_tail Heval)
        as (st1' & Heval' & Hmemories1 & Hcode1 & Hextend & Hextend').
      exists st1'. split; [ exact Heval' | ]. split; [ exact Hmemories1 | ].
      split; [ exact Hcode1 | ].
      split; intros s0 a0 Hlookup;
        [ apply Hextend, Hextend2 | apply Hextend', Hextend2' ];
        exact Hlookup.
Qed.

Lemma states_related_except_intro (D : dropped) (r : correspondence)
    (e : expr) (vars vars' : env) (conts conts' : cont_env)
    (st st' : static_env) (h h' : heap) :
  memories_related D r st st' h h' ->
  static_code st = static_code st' ->
  env_related (value_related false r st st') (fun x => var_free_b x e) vars
    vars' ->
  conts_related (value_related false r st st') conts conts' ->
  states_related_except D (Heap_state e vars conts st h)
    (Heap_state e vars' conts' st' h').
Proof.
  intros Hmemories Hcode Henv Hconts.
  split; [ reflexivity | ]. split; [ exact Hcode | ]. exists r. auto.
Qed.

(* With no symbol left out, it is [states_related]. *)
Theorem states_related_nothing : states_related_nothing_statement.
Proof.
  intros s s'. split.
  - intros (He & Hcode & Hdomain & Himported & r & Hvalid & Hsymbols & Henv
            & Hconts).
    split; [ exact He | ]. split; [ exact Hcode | ]. exists r.
    split; [ | split; assumption ].
    split; [ exact Hvalid | ].
    split; [ intros sym _; apply Hsymbols | ].
    split; [ intros sym _; apply Hdomain | ].
    split; [ intros sym [=] | exact Himported ].
  - intros (He & Hcode & r & (Hvalid & Hsymbols & Hdomain & _ & Himported)
            & Henv & Hconts).
    split; [ exact He | ]. split; [ exact Hcode | ].
    split; [ intros sym; exact (Hdomain sym eq_refl) | ].
    split; [ exact Himported | ]. exists r.
    split; [ exact Hvalid | ].
    split; [ intros sym; exact (Hsymbols sym eq_refl) | ].
    split; assumption.
Qed.

Lemma states_related_intro (r : correspondence) (e : expr) (vars vars' : env)
    (conts conts' : cont_env) (st st' : static_env) (h h' : heap) :
  memories_related nothing_dropped r st st' h h' ->
  static_code st = static_code st' ->
  env_related (value_related false r st st') (fun x => var_free_b x e) vars
    vars' ->
  conts_related (value_related false r st st') conts conts' ->
  states_related (Heap_state e vars conts st h)
    (Heap_state e vars' conts' st' h').
Proof.
  intros Hmemories Hcode Henv Hconts.
  apply states_related_nothing, (states_related_except_intro _ r);
    assumption.
Qed.

(* What a step of the first of two related states is, the step of the
   second is, up to the relation. *)
Definition step_results_related (D : dropped)
    (result result' : heap_step_result) : Prop :=
  match result with
  | Heap_step s1 =>
    ∃ s1' : heap_state,
      result' = Heap_step s1' ∧ states_related_except D s1 s1'
  | Heap_exit k o =>
    ∃ o' : outcome, result' = Heap_exit k o' ∧ outcome_equivalent o o'
  | Heap_undefined => True
  end.

Lemma Forall2_firstn {A B : Type} (R : A -> B -> Prop) (n : nat)
    (l : list A) (l' : list B) :
  Forall2 R l l' -> Forall2 R (firstn n l) (firstn n l').
Proof.
  intros Hall. revert n.
  induction Hall as [ | x y l l' Hxy Hall IH]; intros [ | n]; simpl;
    constructor; auto.
Qed.

Lemma Forall2_skipn {A B : Type} (R : A -> B -> Prop) (n : nat)
    (l : list A) (l' : list B) :
  Forall2 R l l' -> Forall2 R (skipn n l) (skipn n l').
Proof.
  intros Hall. revert n.
  induction Hall as [ | x y l l' Hxy Hall IH]; intros [ | n]; simpl;
    try (constructor; assumption).
  apply IH.
Qed.

(* Related values have the same kinds. *)
Lemma related_kinds (D : dropped) (r : correspondence) (st st' : static_env)
    (h h' : heap) (vs vs' : list value) :
  memories_related D r st st' h h' ->
  Forall2 (value_related false r st st') vs vs' ->
  map kind_of_value vs' = map kind_of_value vs.
Proof.
  intros Hmemories Hvs.
  induction Hvs as [ | v v' vs vs' Hv Hvs IH]; simpl; [ reflexivity | ].
  rewrite IH, (related_kind D r st st' h h' Hmemories _ _ Hv). reflexivity.
Qed.

(* A full call of the same code with related closures, arguments and
   continuations. *)
Lemma related_heap_call_code (D : dropped) (r : correspondence)
    (st st' : static_env) (h h' : heap) (closure closure' : value) (c : code)
    (vs vs' : list value) (args_arity return_arity : arity)
    (ret ret' exn exn' : cont_value) :
  memories_related D r st st' h h' ->
  static_code st = static_code st' ->
  value_related false r st st' closure closure' ->
  Forall2 (value_related false r st st') vs vs' ->
  cont_related (value_related false r st st') ret ret' ->
  cont_related (value_related false r st st') exn exn' ->
  step_results_related D
    (heap_call_code closure c vs args_arity return_arity ret exn st h)
    (heap_call_code closure' c vs' args_arity return_arity ret' exn' st' h').
Proof.
  intros Hmemories Hcode Hclosure Hvs Hret Hexn. unfold heap_call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  rewrite (related_kinds D r st st' h h' _ _ Hmemories Hvs).
  destruct (negb _); [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [env1 | ] eqn:Hbind; [ | exact I ].
  assert (Henv0 :
            env_related (value_related false r st st')
              (fun x => true && negb (among x (code_params c)))
              [(code_my_closure c, closure)] [(code_my_closure c, closure')]).
  { intros y w _. simpl.
    destruct (Pos.eqb y (code_my_closure c)); [ | discriminate ].
    intros [= <-]. exists closure'. auto. }
  destruct
    (related_bind_params _ (fun _ => true) _ _ _ _ _ _ Hvs Henv0 Hbind)
    as (env1' & -> & Henv1).
  eexists. split; [ reflexivity | ].
  apply (states_related_except_intro D r);
    [ exact Hmemories | exact Hcode | | ].
  - revert Henv1. apply env_related_mono; auto.
  - apply Forall2_cons; [ split; [ reflexivity | exact Hret ] | ].
    apply Forall2_cons; [ split; [ reflexivity | exact Hexn ] | ].
    apply Forall2_nil.
Qed.

(* What two calls are, where the second is as the first with related
   values: the same shape, of the same code at the same arities, with
   related values. Nothing is asked where the first is undefined. *)
Definition heap_shapes_related (R : value -> value -> Prop)
    (shape shape' : option call_shape) : Prop :=
  match shape, shape' with
  | None, _ => True
  | Some (Full_call closure c vs args_arity),
    Some (Full_call closure' c' vs' args_arity') =>
    R closure closure' ∧ c' = c ∧ Forall2 R vs vs' ∧ args_arity' = args_arity
  | Some (Over_call closure c vs remaining args_arity remaining_arity),
    Some (Over_call closure' c' vs' remaining' args_arity' remaining_arity')
    =>
    R closure closure'
    ∧ c' = c
    ∧ Forall2 R vs vs'
    ∧ Forall2 R remaining remaining'
    ∧ args_arity' = args_arity
    ∧ remaining_arity' = remaining_arity
  | Some (Partial_call partial), Some (Partial_call partial') =>
    R partial partial'
  | Some _, _ => False
  end.

Lemma related_shape_of_closure_call (D : dropped) (r : correspondence)
    (st st' : static_env) (h h' : heap) (closure closure' : value)
    (kind : call_kind) (vs vs' : list value) (args_arity return_arity : arity) :
  memories_related D r st st' h h' ->
  static_code st = static_code st' ->
  value_related false r st st' closure closure' ->
  Forall2 (value_related false r st st') vs vs' ->
  heap_shapes_related (value_related false r st st')
    (shape_of_closure_call st closure kind vs args_arity return_arity)
    (shape_of_closure_call st' closure' kind vs' args_arity return_arity).
Proof.
  intros Hmemories Hcode Hclosure Hvs. unfold shape_of_closure_call.
  destruct (closure_of st closure) as [[called slots] | ] eqn:Hcalled;
    [ | exact I ].
  destruct
    (related_closure_of D r st st' h h' Hmemories _ _ _ _ Hclosure Hcalled)
    as (slots' & -> & _).
  rewrite <- Hcode.
  destruct
    match kind with
    | Direct named => Pos.eqb called named
    | Indirect => true
    end;
    [ | exact I ].
  destruct (lookup called (static_code st)) as [c | ]; [ | exact I ].
  (* The arguments as the code takes them: for tupled code, the fields
     of two related tuples, which are related. *)
  destruct (arguments_for_code st c vs args_arity)
    as [[ws taken_arity] | ] eqn:Hargs; [ | exact I ].
  destruct
    (FlambdaSemanticsProofs.arguments_for_code_related
       (value_related false r st st') st st' c c vs vs' args_arity ws
       taken_arity (related_respects_observations D r st st' h h' Hmemories)
       eq_refl eq_refl Hvs
       (related_kinds D r st st' h h' _ _ Hmemories Hvs) Hargs)
    as (ws' & -> & Hws).
  cbv beta iota. unfold shape_of_code_call. cbv zeta.
  destruct (Nat.eqb (length taken_arity) (length (code_params_arity c))).
  { simpl. auto. }
  rewrite (related_kinds D r st st' h h' _ _ Hmemories Hws).
  destruct (negb _); [ exact I | ].
  destruct (Nat.ltb (length (code_params_arity c)) (length taken_arity));
    simpl.
  - split; [ exact Hclosure | ]. split; [ reflexivity | ].
    split; [ apply Forall2_firstn; exact Hws | ].
    split; [ apply Forall2_skipn; exact Hws | auto ].
  - destruct (code_is_tupled c); [ exact I | ]. simpl.
    apply Related_step, Step_partial_application; assumption.
Qed.

Lemma related_shape_of_call (D : dropped) (r : correspondence)
    (st st' : static_env) (h h' : heap) (callee callee' : value)
    (kind : call_kind) (vs vs' : list value) (args_arity return_arity : arity) :
  memories_related D r st st' h h' ->
  static_code st = static_code st' ->
  value_related false r st st' callee callee' ->
  Forall2 (value_related false r st st') vs vs' ->
  heap_shapes_related (value_related false r st st')
    (shape_of_call st callee kind vs args_arity return_arity)
    (shape_of_call st' callee' kind vs' args_arity return_arity).
Proof.
  intros Hmemories Hcode Hcallee Hvs. unfold shape_of_call.
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ] eqn:Hpartial.
  - destruct
      (related_partial_application_of D r st st' h h' Hmemories _ _ _ _ _
         Hcallee Hpartial)
      as (closure' & applied' & -> & Hclosure & Happlied).
    simpl.
    destruct kind as [named | ];
      [ apply (related_shape_of_closure_call D r _ _ h h'); assumption | ].
    destruct args_arity as [ | component args_arity]; [ exact I | ].
    rewrite (related_kinds D r st st' h h' _ _ Hmemories Hvs).
    destruct (kinds_eqb _ _); [ | exact I ].
    apply (related_shape_of_closure_call D r _ _ h h'); try assumption.
    apply Forall2_app; assumption.
  - rewrite
      (related_not_partial_application D r st st' h h' Hmemories _ _ Hcallee
         Hpartial).
    destruct kind; apply (related_shape_of_closure_call D r _ _ h h');
      assumption.
Qed.

(* Two calls of related shapes. *)
Lemma related_heap_run_call (D : dropped) (r : correspondence)
    (st st' : static_env) (h h' : heap)
    (return_partial return_partial' : value -> heap_step_result)
    (shape shape' : option call_shape) (return_arity : arity)
    (ret ret' exn exn' : cont_value) :
  memories_related D r st st' h h' ->
  static_code st = static_code st' ->
  (∀ partial partial' : value,
    value_related false r st st' partial partial' ->
    step_results_related D (return_partial partial)
      (return_partial' partial')) ->
  heap_shapes_related (value_related false r st st') shape shape' ->
  cont_related (value_related false r st st') ret ret' ->
  cont_related (value_related false r st st') exn exn' ->
  step_results_related D
    (heap_run_call return_partial shape return_arity ret exn st h)
    (heap_run_call return_partial' shape' return_arity ret' exn' st' h').
Proof.
  intros Hmemories Hcode Hpartial Hshapes Hret Hexn.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ],
    shape'
    as [[closure' c' vs' args_arity'
        | closure' c' vs' remaining' args_arity' remaining_arity'
        | partial'] | ];
    simpl in Hshapes |- *; try contradiction; try exact I.
  - destruct Hshapes as (Hclosure & -> & Hvs & ->).
    apply (related_heap_call_code D r); assumption.
  - destruct Hshapes as (Hclosure & -> & Hvs & Hremaining & -> & ->).
    apply (related_heap_call_code D r); try assumption.
    apply Related_over_application; assumption.
  - apply Hpartial. exact Hshapes.
Qed.

Lemma related_heap_apply_cont_value (D : dropped) (r : correspondence)
    (st st' : static_env) (h h' : heap) (c c' : cont_value)
    (vs vs' : list value) :
  memories_related D r st st' h h' ->
  static_code st = static_code st' ->
  cont_related (value_related false r st st') c c' ->
  Forall2 (value_related false r st st') vs vs' ->
  step_results_related D (heap_apply_cont_value c vs st h)
    (heap_apply_cont_value c' vs' st' h').
Proof.
  intros Hmemories Hcode Hc. revert vs vs'.
  induction Hc
    as [k | handler def_vars def_vars' def_conts def_conts' recursive Henv
              Hconts
       | c c' extra extra' Hc IH Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn IHreturn Hexn IHexn];
    intros vs vs' Hvs.
  4: {
    cbn [heap_apply_cont_value].
    destruct Hvs as [ | v v' vs vs' Hv Hvs]; [ exact I | ].
    destruct Hvs as [ | v2 v2' vs vs' Hv2 Hvs]; [ | exact I ].
    cbn [heap_apply_cont_value].
    apply (related_heap_run_call D r); try assumption.
    - intros partial partial' Hpartial. apply IHreturn.
      constructor; [ exact Hpartial | constructor ].
    - apply (related_shape_of_call D r _ _ h h'); assumption. }
  3: { simpl. apply IH. apply Forall2_app; assumption. }
  - simpl. eexists. split; [ reflexivity | ]. exists r.
    split; [ exact (proj1 Hmemories) | exact Hvs ].
  - destruct handler as [params metadata handler]. simpl.
    assert (Hkinds : map kind_of_value vs' = map kind_of_value vs).
    { clear - Hvs Hmemories.
      induction Hvs as [ | v v' vs vs' Hv Hvs IH]; simpl; [ reflexivity | ].
      rewrite IH, (related_kind D r st st' h h' Hmemories _ _ Hv).
      reflexivity. }
    rewrite Hkinds.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [env1 | ] eqn:Hbind;
      [ | exact I ].
    assert (Henv0 :
              env_related (value_related false r st st')
                (fun x => var_free_b x handler && negb (among x params))
                def_vars def_vars').
    { revert Henv. apply env_related_mono; [ auto | ].
      intros y Hy. unfold needed_by_closure. simpl.
      rewrite andb_comm in Hy. rewrite Hy. reflexivity. }
    destruct
      (related_bind_params _ (fun x => var_free_b x handler) _ _ _ _ _ _ Hvs
         Henv0 Hbind)
      as (env1' & -> & Henv1).
    eexists. split; [ reflexivity | ].
    apply (states_related_except_intro D r);
      [ exact Hmemories | exact Hcode | exact Henv1 | ].
    assert (Henv2 :
              env_related (value_related false r st st')
                (fun x =>
                   existsb (fun '(_, h) => var_free_in_handler_b x h)
                     recursive)
                def_vars def_vars').
    { revert Henv. apply env_related_mono; [ auto | ].
      intros y Hy. unfold needed_by_closure. rewrite Hy. apply orb_true_r. }
    apply Forall2_app; [ | exact Hconts ].
    exact (related_rec_closures _ _ _ _ _ _ Henv2 Hconts).
Qed.

Lemma related_heap_call_function (D : dropped) (r : correspondence)
    (st st' : static_env) (h h' : heap) (callee callee' : value)
    (kind : call_kind) (vs vs' : list value) (args_arity return_arity : arity)
    (ret ret' exn exn' : cont_value) :
  memories_related D r st st' h h' ->
  static_code st = static_code st' ->
  value_related false r st st' callee callee' ->
  Forall2 (value_related false r st st') vs vs' ->
  cont_related (value_related false r st st') ret ret' ->
  cont_related (value_related false r st st') exn exn' ->
  step_results_related D
    (heap_call_function callee kind vs args_arity return_arity ret exn st h)
    (heap_call_function callee' kind vs' args_arity return_arity ret' exn' st'
       h').
Proof.
  intros Hmemories Hcode Hcallee Hvs Hret Hexn. unfold heap_call_function.
  apply (related_heap_run_call D r); try assumption.
  - intros partial partial' Hpartial.
    apply (related_heap_apply_cont_value D r); try assumption.
    constructor; [ exact Hpartial | constructor ].
  - apply (related_shape_of_call D r _ _ h h'); assumption.
Qed.

(* What the next step of an expression names: no symbol left out. *)
Definition head_avoids (D : dropped) (e : expr) : bool :=
  match e with
  | Let _ _ d _ => named_avoids D d
  | Let_static bindings _ => forallb (binding_names_kept D) bindings
  | Apply ap => apply_avoids D ap
  | Apply_cont ac => apply_cont_avoids D ac
  | Switch scrutinee _ => simple_avoids D scrutinee
  | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _ | Invalid => true
  end.

(* One step. *)
Lemma related_heap_step (D : dropped) (s s' : heap_state) :
  states_related_except D s s' ->
  head_avoids D (hs_expr s) = true ->
  step_results_related D (heap_step s) (heap_step s').
Proof.
  destruct s as [e vars conts st h], s' as [e' vars' conts' st' h'].
  intros (He & Hcode & r & Hmemories & Henv & Hconts) Hhead.
  cbn [hs_expr hs_vars hs_conts hs_statics hs_heap]
    in He, Hcode, Hmemories, Henv, Hconts, Hhead.
  subst e'.
  destruct e
    as [x x_kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ]; simpl in Hhead |- *.
  - (* Let *)
    destruct (heap_exec_named st h vars d) as [[v h1] | ] eqn:Hexec;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) x_kind) eqn:Hkind; [ | exact I ].
    assert (Hneeded :
              ∀ y : variable,
                var_in_named y d = true ->
                var_free_b y (Let x x_kind d body) = true).
    { intros y Hy. simpl. rewrite Hy. reflexivity. }
    destruct
      (related_heap_exec_named D r st st' h h' Hmemories _ _ _ _ _ _ Henv
         Hneeded Hhead Hexec)
      as (v' & h1' & r1 & -> & Hmemories1 & Hlocations & Hvalues & Hv).
    rewrite <- (related_kind D r1 st st' h1 h1' Hmemories1 _ _ Hv), Hkind.
    eexists. split; [ reflexivity | ].
    assert (Hgrow :
              ∀ a a' : value,
                value_related false r st st' a a' ->
                value_related false r1 st st' a a').
    { intros a a'. apply value_related_grow; [ exact Hlocations | ].
      rewrite Hvalues. auto. }
    apply (states_related_except_intro D r1);
      [ exact Hmemories1 | exact Hcode | | ].
    + intros y w Hy. cbv beta in Hy. simpl.
      destruct (Pos.eqb_spec y x) as [-> | Hneq].
      * intros [= <-]. exists v'. auto.
      * intros Hw.
        assert (Hfree : var_free_b y (Let x x_kind d body) = true).
        { simpl. rewrite (proj2 (Pos.eqb_neq y x) Hneq). simpl. rewrite Hy.
          apply orb_true_r. }
        destruct (Henv y w Hfree Hw) as (w' & Hw' & Hrelated).
        exists w'. split; [ exact Hw' | exact (Hgrow _ _ Hrelated) ].
    + revert Hconts. apply conts_related_mono. exact Hgrow.
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings st) as [st1 | ] eqn:Heval;
      [ | exact I ].
    assert (Hneeded :
              ∀ y : variable,
                existsb (var_in_static_binding y) bindings = true ->
                var_free_b y (Let_static bindings body) = true).
    { intros y Hy. simpl. rewrite Hy. reflexivity. }
    destruct
      (related_eval_static_bindings D r h h' _ vars vars' bindings st st' st1
         Hmemories Hcode Henv Hneeded Hhead Heval)
      as (st1' & -> & Hmemories1 & Hcode1 & Hextend & Hextend').
    eexists. split; [ reflexivity | ].
    assert (Hmono :
              ∀ a a' : value,
                value_related false r st st' a a' ->
                value_related false r st1 st1' a a').
    { intros a a'.
      apply value_related_statics;
        [ exact Hextend | exact Hextend'
        | exact (memories_opaque D r st st' h h' Hmemories) ]. }
    apply (states_related_except_intro D r);
      [ exact Hmemories1 | exact Hcode1 | | ].
    + revert Henv. apply env_related_mono; [ exact Hmono | ].
      intros y Hy. cbv beta in Hy. simpl. rewrite Hy. apply orb_true_r.
    + revert Hconts. apply conts_related_mono. exact Hmono.
  - (* Let_cont_non_recursive *)
    eexists. split; [ reflexivity | ].
    apply (states_related_except_intro D r);
      [ exact Hmemories | exact Hcode | | ].
    + revert Henv. apply env_related_mono; [ auto | ].
      intros y Hy. cbv beta in Hy. simpl. rewrite Hy. apply orb_true_r.
    + apply Forall2_cons; [ | exact Hconts ]. split; [ reflexivity | ].
      simpl. apply Related_closure; [ | exact Hconts ].
      revert Henv. apply env_related_mono; [ auto | ].
      intros y Hy. unfold needed_by_closure in Hy. simpl in Hy.
      rewrite orb_false_r in Hy. simpl. rewrite Hy. reflexivity.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ exact I | ].
    eexists. split; [ reflexivity | ].
    apply (states_related_except_intro D r);
      [ exact Hmemories | exact Hcode | | ].
    + revert Henv. apply env_related_mono; [ auto | ].
      intros y Hy. cbv beta in Hy. simpl. rewrite Hy. apply orb_true_r.
    + assert (Henv2 :
                env_related (value_related false r st st')
                  (fun x =>
                     existsb (fun '(_, h) => var_free_in_handler_b x h)
                       handlers)
                  vars vars').
      { revert Henv. apply env_related_mono; [ auto | ].
        intros y Hy. simpl. rewrite Hy. reflexivity. }
      apply Forall2_app; [ | exact Hconts ].
      exact (related_rec_closures _ _ _ _ _ _ Henv2 Hconts).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra; [ | exact I ].
    assert (Hneeded1 :
              ∀ y : variable,
                var_in_simple y (ap_callee ap) = true ->
                var_free_b y (Apply ap) = true).
    { intros y Hy. simpl. rewrite Hy. reflexivity. }
    assert (Hneeded2 :
              ∀ y : variable,
                var_in_simples y (ap_args ap) = true ->
                var_free_b y (Apply ap) = true).
    { intros y Hy. simpl. rewrite Hy, orb_true_r. reflexivity. }
    assert (Hneeded3 :
              ∀ y : variable,
                var_in_simples y (exn_extra_simples ap) = true ->
                var_free_b y (Apply ap) = true).
    { intros y Hy. simpl. rewrite Hy. apply orb_true_r. }
    apply andb_true_iff in Hhead as [Hhead Havoid_extra].
    apply andb_true_iff in Hhead as [Hhead Havoid_args].
    apply andb_true_iff in Hhead as [Havoid_callee _].
    destruct
      (related_eval_simple D r st st' h h' Hmemories _ _ _ _ _ Henv Hneeded1
         Havoid_callee Hcallee)
      as (callee' & -> & Hcallee').
    destruct
      (related_eval_simples D r st st' h h' Hmemories _ _ _ _ _ Henv Hneeded2
         Havoid_args Hvs)
      as (vs' & -> & Hvs').
    destruct
      (related_eval_simples D r st st' h h' Hmemories _ _ _ _ _ Henv Hneeded3
         Havoid_extra Hextra)
      as (extra' & -> & Hextra').
    apply (related_heap_call_function D r); try assumption.
    + apply related_resolve; exact Hconts.
    + apply related_with_extra_args; [ | exact Hextra' ].
      apply related_resolve; exact Hconts.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    assert (Hneeded :
              ∀ y : variable,
                var_in_simples y (ac_args ac) = true ->
                var_free_b y (Apply_cont ac) = true).
    { intros y Hy. exact Hy. }
    destruct
      (related_eval_simples D r st st' h h' Hmemories _ _ _ _ _ Henv Hneeded
         Hhead Hvs)
      as (vs' & -> & Hvs').
    apply (related_heap_apply_cont_value D r); try assumption.
    apply related_resolve. exact Hconts.
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
    assert (Hneeded :
              ∀ y : variable,
                var_in_simple y scrutinee = true ->
                var_free_b y (Switch scrutinee arms) = true).
    { intros y Hy. simpl. rewrite Hy. reflexivity. }
    destruct
      (related_eval_simple D r st st' h h' Hmemories _ _ _ _ _ Henv Hneeded
         Hhead Hv)
      as (v' & -> & Hv').
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
    assert (Hsame : v' = v).
    { destruct v as [c | | | | | | | | ]; try discriminate.
      exact (related_const_left D r st st' h h' Hmemories c v' Hv'). }
    subst v'. rewrite Hi.
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    eexists. split; [ reflexivity | ].
    apply (states_related_except_intro D r);
      [ exact Hmemories | exact Hcode | | exact Hconts ].
    revert Henv. apply env_related_mono; [ auto | ].
    intros y Hy. cbv beta in Hy. simpl in Hy |- *.
    destruct (find_arm_in _ _ _ Harm) as (d0 & Hin).
    apply orb_true_iff. right. apply existsb_exists. exists (d0, ac).
    split; [ exact Hin | exact Hy ].
  - (* Invalid *)
    exact I.
Qed.

(* The simulation, up to symbols left out, for the states of an invariant
   that steps keep and under which the next step names no symbol left
   out. *)
Section Simulation.

Variable D : dropped.
Variable invariant : heap_state -> Prop.
Hypothesis invariant_step :
  ∀ s s1 : heap_state,
    invariant s -> heap_step s = Heap_step s1 -> invariant s1.
Hypothesis invariant_head :
  ∀ s : heap_state, invariant s -> head_avoids D (hs_expr s) = true.

Lemma heap_run_related_except (fuel : nat) :
  ∀ s s' : heap_state,
    invariant s ->
    states_related_except D s s' ->
    match heap_run fuel s with
    | Heap_step s1 =>
      ∃ s1' : heap_state,
        heap_run fuel s' = Heap_step s1' ∧ states_related_except D s1 s1'
    | Heap_exit k o =>
      ∃ o' : outcome,
        heap_run fuel s' = Heap_exit k o' ∧ outcome_equivalent o o'
    | Heap_undefined => True
    end.
Proof.
  induction fuel as [ | fuel IH]; intros s s' Hinvariant Hrelated; simpl.
  - exists s'. auto.
  - pose proof (related_heap_step D s s' Hrelated (invariant_head s Hinvariant))
      as Hstep.
    pose proof (invariant_step s) as Hkeeps.
    destruct (heap_step s) as [s1 | k o | ]; simpl in Hstep.
    + destruct Hstep as (s1' & -> & Hrelated1).
      apply IH; [ exact (Hkeeps s1 Hinvariant eq_refl) | exact Hrelated1 ].
    + destruct Hstep as (o' & -> & Ho). exists o'. auto.
    + exact I.
Qed.

Lemma heap_refines_except (s s' : heap_state) :
  invariant s -> states_related_except D s s' -> heap_refines s s'.
Proof.
  intros Hinvariant Hrelated. split.
  - intros k o [fuel Hrun].
    pose proof (heap_run_related_except fuel s s' Hinvariant Hrelated)
      as Hsimulation.
    rewrite Hrun in Hsimulation. destruct Hsimulation as (o' & Hrun' & Ho).
    exists o'.
    split;
      [ exists fuel; exact Hrun'
      | apply outcome_equivalent_refines; exact Ho ].
  - intros b' Hnot Hb'.
    destruct b' as [k o' | | ]; [ | | exfalso; apply Hnot; reflexivity ];
      destruct Hb' as [fuel Hrun'];
      pose proof (heap_run_related_except fuel s s' Hinvariant Hrelated)
        as Hsimulation;
      destruct (heap_run fuel s) as [s1 | k1 o | ] eqn:Hrun.
    + destruct Hsimulation as (s1' & Hs1' & _). rewrite Hrun' in Hs1'.
      discriminate.
    + destruct Hsimulation as (o1' & Ho1' & Ho). rewrite Hrun' in Ho1'.
      injection Ho1' as <- <-. left. exists (Heap_exits k o).
      split; [ exists fuel; exact Hrun | ].
      split; [ reflexivity | apply outcome_equivalent_refines; exact Ho ].
    + right. exists fuel. exact Hrun.
    + destruct Hsimulation as (s1' & Hs1' & _). rewrite Hrun' in Hs1'.
      discriminate.
    + destruct Hsimulation as (o1' & Ho1' & _). rewrite Hrun' in Ho1'.
      discriminate.
    + right. exists fuel. exact Hrun.
Qed.

End Simulation.

(* With no symbol left out, every expression names none. *)
Lemma simple_avoids_nothing (a : simple) :
  simple_avoids nothing_dropped a = true.
Proof. destruct a; reflexivity. Qed.

Lemma simples_avoid_nothing (args : list simple) :
  simples_avoid nothing_dropped args = true.
Proof. apply forallb_forall. intros a _. apply simple_avoids_nothing. Qed.

Lemma value_slots_avoid_nothing (slots : list (value_slot * simple)) :
  value_slots_avoid nothing_dropped slots = true.
Proof. apply forallb_forall. intros a _. apply simple_avoids_nothing. Qed.

Lemma head_avoids_nothing (e : expr) : head_avoids nothing_dropped e = true.
Proof.
  destruct e
    as [x x_kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ]; simpl; try reflexivity.
  - destruct d; simpl;
      rewrite ?simple_avoids_nothing, ?simples_avoid_nothing,
        ?value_slots_avoid_nothing;
      try reflexivity.
    rewrite andb_true_r. apply forallb_forall. intros decl _. reflexivity.
  - apply forallb_forall. intros [id c | id | s const] _; try reflexivity.
    destruct const; simpl;
      rewrite ?simple_avoids_nothing, ?simples_avoid_nothing,
        ?value_slots_avoid_nothing;
      try reflexivity.
    rewrite andb_true_r. apply andb_true_iff.
    split; apply forallb_forall; intros entry _; reflexivity.
  - unfold apply_avoids.
    rewrite simple_avoids_nothing, !simples_avoid_nothing.
    destruct (ap_call_kind ap); reflexivity.
  - apply simples_avoid_nothing.
  - apply simple_avoids_nothing.
Qed.

Theorem heap_run_related : heap_run_related_statement.
Proof.
  intros fuel s s' Hrelated.
  apply states_related_nothing in Hrelated.
  pose proof
    (heap_run_related_except nothing_dropped (fun _ => True)
       (fun _ _ _ _ => I) (fun s0 _ => head_avoids_nothing (hs_expr s0)) fuel
       s s' I Hrelated)
    as Hsimulation.
  destruct (heap_run fuel s) as [s1 | k o | ];
    [ | exact Hsimulation | exact I ].
  destruct Hsimulation as (s1' & Hrun' & Hrelated1). exists s1'.
  split; [ exact Hrun' | apply states_related_nothing; exact Hrelated1 ].
Qed.

Theorem machine_respects_relation : machine_respects_relation_statement.
Proof.
  intros s s' Hrelated.
  apply
    (heap_refines_except nothing_dropped (fun _ => True) (fun _ _ _ _ => I)
       (fun s0 _ => head_avoids_nothing (hs_expr s0)));
    [ exact I | apply states_related_nothing; exact Hrelated ].
Qed.

(* ======================================================================== *)
(* Closed states, and the rewrites that follow                              *)
(* ======================================================================== *)

(* The correspondence that is the identity on the locations of a heap,
   and under which every symbol that is not left out is itself. *)
Definition identity_correspondence_except (D : dropped) (h : heap)
    : correspondence :=
  Correspondence
    (fun l l' => l = l' ∧ is_bound l h = true)
    (fun v v' =>
       ∃ s : symbol, symbol_kept D s = true ∧ v = Value_symbol s ∧ v' = v).

Definition identity_correspondence (h : heap) : correspondence :=
  identity_correspondence_except nothing_dropped h.

Lemma closed_value_related_except (D : dropped) (st st1 : static_env)
    (h : heap) (v : value) :
  value_closed h v ->
  value_avoids D v = true ->
  value_related false (identity_correspondence_except D h) st st1 v v.
Proof.
  intros Hclosed Havoids. apply (value_related_same D).
  - intros s Hkept. exists s. auto.
  - exact Havoids.
  - intros l Hl. split; [ reflexivity | exact (Hclosed l Hl) ].
Qed.

Lemma closed_value_related (st : static_env) (h : heap) (v : value) :
  value_closed h v ->
  value_related false (identity_correspondence h) st st v v.
Proof.
  intros Hclosed.
  apply closed_value_related_except;
    [ exact Hclosed | apply value_avoids_nothing ].
Qed.

(* The static sets of closures of a static environment name no symbol
   left out. *)
Definition static_sets_kept (D : dropped) (st : static_env) : Prop :=
  ∀ (s : symbol) (slot : function_slot)
    (symbols : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)),
    lookup s (static_values st)
    = Some (Static_closure_of_set slot symbols decls slots) ->
    closure_symbols_avoid D symbols = true.

(* The projection of a function slot from a symbol of a static set is a
   symbol that the set names. *)
Lemma project_symbol_of_set (st : static_env) (s : symbol)
    (slot : function_slot) (syms : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) (move_from move_to : function_slot)
    (w : value) :
  lookup s (static_values st)
  = Some (Static_closure_of_set slot syms decls slots) ->
  project_function_slot st move_from move_to (Value_symbol s) = Some w ->
  ∃ t : symbol, w = Value_symbol t ∧ (move_to, t) ∈ syms.
Proof.
  intros Hlookup. cbn -[Pos.eqb]. rewrite Hlookup.
  destruct (Pos.eqb slot move_from); [ | discriminate ].
  destruct (lookup slot decls); simpl; [ | discriminate ].
  destruct (lookup move_to decls); simpl; [ | discriminate ].
  destruct (lookup move_to syms) as [t | ] eqn:Ht; simpl; [ | discriminate ].
  intros [= <-]. exists t. split; [ reflexivity | exact (lookup_in _ _ _ Ht) ].
Qed.

(* A symbol bound to the same closure of a static set in two static
   environments projects to the same symbols, which are related to
   themselves where the set names no symbol left out. *)
Lemma static_set_projections (D : dropped) (st st1 : static_env) (h : heap)
    (s : symbol) (slot : function_slot)
    (syms : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  lookup s (static_values st)
  = Some (Static_closure_of_set slot syms decls slots) ->
  lookup s (static_values st1)
  = Some (Static_closure_of_set slot syms decls slots) ->
  closure_symbols_avoid D syms = true ->
  (∀ (move_to : function_slot) (w : value),
    project_function_slot st slot move_to (Value_symbol s) = Some w ->
    ∃ w' : value,
      project_function_slot st1 slot move_to (Value_symbol s) = Some w'
      ∧ value_related false (identity_correspondence_except D h) st st1 w w')
  ∧ (∀ move_to : function_slot,
      project_function_slot st slot move_to (Value_symbol s) = None ->
      project_function_slot st1 slot move_to (Value_symbol s) = None).
Proof.
  intros Hlookup Hlookup1 Hsyms. split.
  2: { intros move_to Hnone. unfold project_function_slot in Hnone |- *.
       rewrite Hlookup1. rewrite Hlookup in Hnone. exact Hnone. }
  intros move_to w Hw.
  destruct (project_symbol_of_set _ _ _ _ _ _ _ _ _ Hlookup Hw)
    as (t & -> & Hin).
  exists (Value_symbol t). split.
  - unfold project_function_slot in Hw |- *. rewrite Hlookup1.
    rewrite Hlookup in Hw. exact Hw.
  - apply Related_corr. exists t. split; [ | auto ].
    unfold closure_symbols_avoid in Hsyms. rewrite forallb_forall in Hsyms.
    exact (Hsyms (move_to, t) Hin).
Qed.

Lemma is_allocated_same_lookup (st st1 : static_env) (s : symbol) :
  lookup s (static_values st1) = lookup s (static_values st) ->
  is_allocated st1 (Value_symbol s) = is_allocated st (Value_symbol s).
Proof.
  intros Hsame.
  unfold is_allocated, closure_of, block_of, boxed_number_of, string_of,
    array_of.
  rewrite Hsame. reflexivity.
Qed.

(* A closed memory that names no symbol left out is related to itself
   with more cells and with the symbols left out bound, to things that
   are allocated. *)
Lemma closed_memories_related_except (D : dropped) (st st1 : static_env)
    (h h1 : heap) :
  memory_closed st h ->
  (∀ (l : location) (c : cell), lookup l h = Some c -> lookup l h1 = Some c) ->
  (∀ (l : location) (c : cell) (v : value),
    lookup l h = Some c -> v ∈ values_of_cell c -> value_avoids D v = true) ->
  (∀ (s : symbol) (a : statically_allocated_value) (v : value),
    lookup s (static_values st) = Some a ->
    v ∈ values_of_static a ->
    value_avoids D v = true) ->
  static_sets_kept D st ->
  (∀ s : symbol,
    symbol_kept D s = true ->
    lookup s (static_values st1) = lookup s (static_values st)) ->
  (∀ s : symbol,
    symbol_kept D s = false ->
    lookup s (static_values st) = None
    ∧ is_allocated st1 (Value_symbol s) = true) ->
  memories_related D (identity_correspondence_except D h) st st1 h1 h.
Proof.
  intros [Hheap Hstatics] Hkept Hheap_avoids Hstatics_avoid Hsets Hagree
    Hleft_out.
  split; [ split; [ | split ] | split; [ | split; [ | split ] ] ]; simpl.
  - intros a a' b b' [-> _] [-> _]. reflexivity.
  - intros l l' [<- Hbound]. unfold is_bound in Hbound.
    destruct (lookup l h) as [c | ] eqn:Hlookup; [ | discriminate ].
    exists c, c. split; [ exact (Hkept _ _ Hlookup) | ].
    split; [ reflexivity | ].
    destruct c as [mut tag fields | kind elements]; simpl.
    + split; [ reflexivity | ]. split; [ reflexivity | ].
      apply Forall2_same. intros field Hfield.
      apply closed_value_related_except.
      * exact (proj1 (Hheap l _ field Hlookup Hfield)).
      * exact (Hheap_avoids l _ field Hlookup Hfield).
    + split; [ reflexivity | ].
      apply Forall2_same. intros element Helement.
      apply closed_value_related_except.
      * exact (proj1 (Hheap l _ element Hlookup Helement)).
      * exact (Hheap_avoids l _ element Hlookup Helement).
  - intros v v' (s & Hkept_s & -> & ->).
    pose proof (Hagree s Hkept_s) as Hsame.
    destruct (lookup s (static_values st)) as [a | ] eqn:Hlookup.
    + pose proof (Hstatics s a) as Hclosed.
      pose proof (Hstatics_avoid s a) as Havoids.
      destruct a
        as [code slots | slot syms decls slots | tag fields | contents
           | bytes | elements | ].
      * apply
          (Step_closure _ _ _ _ _ _ _ unnamed_function_slot
             [(unnamed_function_slot, code)] slots slots);
          [ simpl; rewrite Hlookup; reflexivity
          | simpl; rewrite Hsame; reflexivity | reflexivity
          | | left; split; unfold set_bound; rewrite ?Hsame, ?Hlookup;
              exact I ].
        apply Forall2_same. intros v Hv. apply closed_value_related_except.
        -- exact (Hclosed v Hlookup Hv).
        -- exact (Havoids v Hlookup Hv).
      * destruct (lookup slot decls) as [code | ] eqn:Hslot.
        -- apply (Step_closure _ _ _ _ _ _ _ slot decls slots slots);
             [ simpl; rewrite Hlookup; simpl; rewrite Hslot; reflexivity
             | simpl; rewrite Hsame; simpl; rewrite Hslot; reflexivity
             | reflexivity
             | | right;
                 exact
                   (static_set_projections D st st1 h s slot syms decls slots
                      Hlookup Hsame (Hsets _ _ _ _ _ Hlookup)) ].
           apply Forall2_same. intros v Hv.
           apply closed_value_related_except.
           ++ exact (Hclosed v Hlookup Hv).
           ++ exact (Havoids v Hlookup Hv).
        -- apply Step_opaque_symbol; unfold is_allocated; simpl;
             rewrite ?Hsame, ?Hlookup, Hslot; reflexivity.
      * apply (Step_block _ _ _ _ _ _ _ tag fields fields);
          simpl; rewrite ?Hsame, ?Hlookup; try reflexivity.
        apply Forall2_same. intros v Hv. apply closed_value_related_except.
        -- exact (Hclosed v Hlookup Hv).
        -- exact (Havoids v Hlookup Hv).
      * assert (Hboxed1 :
                  boxed_number_of st1 (Value_symbol s)
                  = boxed_number_of st (Value_symbol s)).
        { simpl. rewrite Hsame, Hlookup. reflexivity. }
        destruct (boxed_number_of st (Value_symbol s)) as [c | ] eqn:Hboxed.
        -- exact
             (Step_boxed_number _ _ _ _ _ _ _ c c Hboxed Hboxed1
                (or_introl eq_refl)).
        -- assert (Hnot : is_allocated st (Value_symbol s) = false).
           { unfold is_allocated. rewrite Hboxed. simpl. rewrite Hlookup.
             reflexivity. }
           apply Step_opaque_symbol; [ exact Hnot | ].
           rewrite (is_allocated_same_lookup st st1);
             [ exact Hnot | rewrite Hlookup; exact Hsame ].
      * apply (Step_string _ _ _ _ _ _ _ bytes); simpl;
          rewrite ?Hsame, ?Hlookup; reflexivity.
      * apply (Step_array _ _ _ _ _ _ _ elements elements);
          simpl; rewrite ?Hsame, ?Hlookup; try reflexivity.
        apply Forall2_same. intros v Hv. apply closed_value_related_except.
        -- exact (Hclosed v Hlookup Hv).
        -- exact (Havoids v Hlookup Hv).
      * assert (Hnot : is_allocated st (Value_symbol s) = false).
        { unfold is_allocated. simpl. rewrite Hlookup. reflexivity. }
        apply Step_opaque_symbol; [ exact Hnot | ].
        rewrite (is_allocated_same_lookup st st1);
          [ exact Hnot | rewrite Hlookup; exact Hsame ].
    + assert (Hnot : is_allocated st (Value_symbol s) = false).
      { unfold is_allocated. simpl. rewrite Hlookup. reflexivity. }
      apply Step_opaque_symbol; [ exact Hnot | ].
      rewrite (is_allocated_same_lookup st st1);
        [ exact Hnot | rewrite Hlookup; exact Hsame ].
  - intros s Hkept_s. exists s. auto.
  - intros s Hkept_s. unfold is_bound. rewrite (Hagree s Hkept_s).
    reflexivity.
  - exact Hleft_out.
  - intros s c.
    destruct (symbol_kept D s) eqn:Hkept_s;
      [ rewrite (Hagree s Hkept_s); split; exact (fun H => H) | ].
    destruct (Hleft_out s Hkept_s) as [Hnone Hallocated].
    rewrite Hnone. split; [ discriminate | ].
    intros Hs. unfold is_allocated in Hallocated. simpl in Hallocated.
    rewrite Hs in Hallocated. discriminate.
Qed.

(* A closed memory is related to itself, and to itself with more cells. *)
Lemma closed_memories_related (st : static_env) (h h1 : heap) :
  memory_closed st h ->
  (∀ (l : location) (c : cell), lookup l h = Some c -> lookup l h1 = Some c) ->
  memories_related nothing_dropped (identity_correspondence h) st st h1 h.
Proof.
  intros Hmemory Hkept.
  apply closed_memories_related_except;
    [ exact Hmemory | exact Hkept | | | | | ].
  - intros l c v _ _. apply value_avoids_nothing.
  - intros s a v _ _. apply value_avoids_nothing.
  - intros s slot syms decls slots _. apply forallb_forall.
    intros entry _. reflexivity.
  - intros s _. reflexivity.
  - intros s [=].
Qed.

Lemma closed_cont_related (R : value -> value -> Prop) (h : heap)
    (c : cont_value) :
  (∀ v : value, value_closed h v -> R v v) ->
  cont_value_closed h c ->
  cont_related R c c.
Proof.
  intros Hsame.
  induction c
    as [handler vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_induction; simpl.
  4: { intros (Hremaining & Hreturn & Hexn).
       apply Related_over_application;
         [ | exact (IHreturn Hreturn) | exact (IHexn Hexn) ].
       apply Forall2_same. intros v Hv. apply Hsame.
       exact (Hremaining v Hv). }
  3: { intros [Hextra Hc]. apply Related_extra_args; [ exact (IH Hc) | ].
       apply Forall2_same. intros v Hv. apply Hsame. exact (Hextra v Hv). }
  - intros [Hvars Hconts]. apply Related_closure.
    + intros x v _ Hv. exists v. split; [ exact Hv | ].
      apply Hsame. exact (Hvars x v (lookup_in _ _ _ Hv)).
    + revert IH Hconts.
      induction conts as [ | [k c] conts IHconts]; intros IH Hconts;
        [ constructor | ].
      destruct Hconts as [Hc Hrest].
      inversion IH as [ | b l Hb Hl]; subst.
      constructor;
        [ split; [ reflexivity | exact (Hb Hc) ] | exact (IHconts Hl Hrest) ].
  - intros _. apply Related_outer.
Qed.

Lemma closed_conts_related (R : value -> value -> Prop) (h : heap)
    (conts : cont_env) :
  (∀ v : value, value_closed h v -> R v v) ->
  (∀ (k : continuation) (c : cont_value),
    (k, c) ∈ conts -> cont_value_closed h c) ->
  conts_related R conts conts.
Proof.
  intros Hsame Hconts. apply Forall2_same. intros [k c] Hin.
  split; [ reflexivity | ].
  exact (closed_cont_related R h c Hsame (Hconts k c Hin)).
Qed.

Theorem closed_state_related : closed_state_related_statement.
Proof.
  intros [e vars conts st h] (Hmemory & Hvars & Hconts).
  cbn [hs_expr hs_vars hs_conts hs_statics hs_heap] in Hmemory, Hvars, Hconts.
  apply (states_related_intro (identity_correspondence h)).
  - apply closed_memories_related; [ exact Hmemory | auto ].
  - reflexivity.
  - intros x v _ Hv. exists v. split; [ exact Hv | ].
    apply closed_value_related. exact (Hvars x v (lookup_in _ _ _ Hv)).
  - apply (closed_conts_related _ h); [ | exact Hconts ].
    intros v. apply closed_value_related.
Qed.

Theorem heap_refines_reflexive : heap_refines_reflexive_statement.
Proof.
  intros s Hclosed. apply machine_respects_relation, closed_state_related.
  exact Hclosed.
Qed.

(* Refinement, one step later. *)
Lemma heap_step_let (x : variable) (x_kind : flambda_kind) (d : named)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (h : heap) :
  heap_step (Heap_state (Let x x_kind d body) vars conts statics h)
  = match heap_exec_named statics h vars d with
    | Some (v, h') =>
      if kind_eqb (kind_of_value v) x_kind
      then Heap_step (Heap_state body ((x, v) :: vars) conts statics h')
      else Heap_undefined
    | None => Heap_undefined
    end.
Proof. reflexivity. Qed.

Lemma heap_run_after_step (s s1 : heap_state) (fuel : nat) :
  heap_step s = Heap_step s1 -> heap_run (S fuel) s = heap_run fuel s1.
Proof. intros Hstep. simpl. rewrite Hstep. reflexivity. Qed.

Lemma heap_behaves_after_step (s s1 : heap_state) (b : heap_behaviour) :
  heap_step s = Heap_step s1 ->
  b <> Heap_diverges ->
  heap_behaves s b <-> heap_behaves s1 b.
Proof.
  intros Hstep Hnot.
  destruct b as [k o | | ]; [ | | exfalso; apply Hnot; reflexivity ];
    split; intros [fuel Hrun].
  - destruct fuel as [ | fuel]; [ discriminate | ]. exists fuel.
    rewrite <- (heap_run_after_step _ _ _ Hstep). exact Hrun.
  - exists (S fuel). rewrite (heap_run_after_step _ _ _ Hstep). exact Hrun.
  - destruct fuel as [ | fuel]; [ discriminate | ]. exists fuel.
    rewrite <- (heap_run_after_step _ _ _ Hstep). exact Hrun.
  - exists (S fuel). rewrite (heap_run_after_step _ _ _ Hstep). exact Hrun.
Qed.

Lemma refines_not_diverges (b b' : heap_behaviour) :
  heap_behaviour_refines b b' -> b' <> Heap_diverges -> b <> Heap_diverges.
Proof.
  intros Hrefines Hnot ->. destruct b'; simpl in Hrefines; contradiction.
Qed.

Lemma heap_refines_after_source_step (s s1 t : heap_state) :
  heap_step s = Heap_step s1 -> heap_refines s1 t -> heap_refines s t.
Proof.
  intros Hstep [Hexits Hback]. split.
  - intros k o Hb. apply Hexits.
    apply (heap_behaves_after_step s s1 _ Hstep); [ discriminate | exact Hb ].
  - intros b' Hnot Hb'.
    destruct (Hback b' Hnot Hb') as [(b & Hb & Hrefines) | Hwrong].
    + left. exists b. split; [ | exact Hrefines ].
      apply (heap_behaves_after_step s s1 b Hstep);
        [ exact (refines_not_diverges _ _ Hrefines Hnot) | exact Hb ].
    + right.
      apply (heap_behaves_after_step s s1 _ Hstep);
        [ discriminate | exact Hwrong ].
Qed.

Lemma heap_refines_after_target_step (s t t1 : heap_state) :
  heap_step t = Heap_step t1 -> heap_refines s t1 -> heap_refines s t.
Proof.
  intros Hstep [Hexits Hback]. split.
  - intros k o Hb. destruct (Hexits k o Hb) as (o' & Hb' & Ho).
    exists o'. split; [ | exact Ho ].
    apply (heap_behaves_after_step t t1 _ Hstep); [ discriminate | exact Hb' ].
  - intros b' Hnot Hb'. apply Hback; [ exact Hnot | ].
    apply (heap_behaves_after_step t t1 b' Hstep); [ exact Hnot | exact Hb' ].
Qed.

(* A state that is undefined at once is refined by anything. *)
Lemma heap_refines_undefined (s t : heap_state) :
  heap_step s = Heap_undefined -> heap_refines s t.
Proof.
  intros Hstep.
  assert (Hwrong : heap_behaves s Heap_goes_wrong).
  { exists 1%nat. simpl. rewrite Hstep. reflexivity. }
  split.
  - intros k o [fuel Hrun]. destruct fuel as [ | fuel]; [ discriminate | ].
    simpl in Hrun. rewrite Hstep in Hrun. discriminate.
  - intros b' _ _. right. exact Hwrong.
Qed.

(* A load from an immutable block that the environment knows. *)
Theorem load_known_field : load_known_field_statement.
Proof.
  intros x b x_kind kind mut field known body vars conts statics h block
    loaded Hblock Hload Hknown Hclosed.
  pose (s1 := Heap_state body ((x, loaded) :: vars) conts statics h).
  pose proof Hload as Hfield. rewrite eval_block_load_load_field in Hfield.
  destruct (block_of statics block) as [[tag fields] | ] eqn:Hblock_of;
    simpl in Hfield; [ | discriminate ].
  assert (Hlocation : location_of block = None).
  { destruct block; try reflexivity. discriminate. }
  assert (Hbound : forallb (argument_bound statics h) [block] = true).
  { destruct block; try reflexivity; try discriminate. simpl in Hblock_of |- *.
    unfold is_bound. destruct (lookup s (static_values statics));
      [ reflexivity | discriminate ]. }
  assert (Hexec :
            heap_exec_named statics h vars
              (Prim (Unary (Block_load kind mut field)) [Var b])
            = Some (loaded, h)).
  { cbn [heap_exec_named eval_simples eval_simple]. rewrite Hblock.
    change
      (exec_prim statics h (Unary (Block_load kind mut field)) [block]
       = Some (loaded, h)).
    unfold exec_prim. rewrite Hbound, exec_bound_prim_unary, Hlocation.
    change (eval_unary_primitive statics (Block_load kind mut field) block)
      with (eval_block_load statics kind field block).
    rewrite Hload. reflexivity. }
  destruct (kind_eqb (kind_of_value loaded) x_kind) eqn:Hkind.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hexec, Hkind.
       reflexivity. }
  apply (heap_refines_after_source_step _ s1);
    [ rewrite heap_step_let, Hexec, Hkind; reflexivity | ].
  apply (heap_refines_after_target_step _ _ s1).
  { rewrite heap_step_let. cbn [heap_exec_named eval_named]. rewrite Hknown.
    simpl. rewrite Hkind. reflexivity. }
  apply heap_refines_reflexive.
  destruct Hclosed as (Hmemory & Hvars & Hconts).
  cbn [hs_expr hs_vars hs_conts hs_statics hs_heap] in Hmemory, Hvars, Hconts.
  split; [ exact Hmemory | split; [ | exact Hconts ] ].
  cbn [hs_vars hs_heap s1].
  intros y v [[= <- <-] | Hin]; [ | exact (Hvars y v Hin) ].
  apply load_field_in in Hfield. intros l Hl.
  destruct block; simpl in Hblock_of; try discriminate.
  - destruct (lookup s (static_values statics))
      as [[ | | tag0 fields0 | | | | ] | ] eqn:Hs; try discriminate.
    injection Hblock_of as <- <-.
    exact (proj2 Hmemory s _ loaded Hs Hfield l Hl).
  - injection Hblock_of as <- <-.
    apply (Hvars b _ (lookup_in _ _ _ Hblock)).
    simpl. apply in_flat_map. exists loaded. auto.
Qed.

(* A binding that is not used and has at most generative effects. *)
Theorem delete_unused_let : delete_unused_let_statement.
Proof.
  intros x x_kind p args body Heffects Hunused vars conts statics h Hclosed.
  destruct (heap_exec_named statics h vars (Prim p args)) as [[v h1] | ]
    eqn:Hexec.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hexec.
       reflexivity. }
  destruct (kind_eqb (kind_of_value v) x_kind) eqn:Hkind.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hexec, Hkind.
       reflexivity. }
  apply
    (heap_refines_after_source_step _
       (Heap_state body ((x, v) :: vars) conts statics h1));
    [ rewrite heap_step_let, Hexec, Hkind; reflexivity | ].
  apply machine_respects_relation.
  destruct Hclosed as (Hmemory & Hvars & Hconts).
  cbn [hs_expr hs_vars hs_conts hs_statics hs_heap] in Hmemory, Hvars, Hconts.
  assert (Hkept :
            ∀ (l : location) (c : cell),
              lookup l h = Some c -> lookup l h1 = Some c).
  { cbn [heap_exec_named] in Hexec.
    destruct (eval_simples vars args) as [vs | ]; [ | discriminate ].
    change (exec_prim statics h p vs = Some (v, h1)) in Hexec.
    unfold at_most_generative_effects in Heffects.
    destruct (effects_of_prim p) as [ | mut | ] eqn:Heff; [ | | discriminate ].
    - rewrite (no_effects _ _ _ _ _ _ Heff Hexec). auto.
    - destruct (only_generative_effects _ _ _ _ _ _ _ Heff Hexec)
        as [[_ ->] | (_ & _ & c0 & ->)]; [ auto | ].
      intros l c Hl. apply lookup_fresh_location. exact Hl. }
  apply (states_related_intro (identity_correspondence h)).
  - apply closed_memories_related; assumption.
  - reflexivity.
  - intros y w Hy. cbv beta in Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _];
      [ rewrite Hunused in Hy; discriminate | ].
    intros Hw. exists w. split; [ exact Hw | ].
    apply closed_value_related. exact (Hvars y w (lookup_in _ _ _ Hw)).
  - apply (closed_conts_related _ h); [ | exact Hconts ].
    intros w. apply closed_value_related.
Qed.

(* ======================================================================== *)
(* A step keeps a state closed                                              *)
(* ======================================================================== *)

(* The continuations that a closure holds, closed. *)
Fixpoint conts_closed (h : heap) (conts : list (continuation * cont_value))
    : Prop :=
  match conts with
  | [] => True
  | (_, c) :: conts => cont_value_closed h c ∧ conts_closed h conts
  end.

Lemma cont_value_closed_closure (h : heap) (handler : continuation_handler)
    (vars : env) (conts : list (continuation * cont_value))
    (recursive : list (continuation * continuation_handler)) :
  cont_value_closed h (Closure handler vars conts recursive)
  <-> (∀ (x : variable) (v : value), (x, v) ∈ vars -> value_closed h v)
      ∧ conts_closed h conts.
Proof.
  simpl. split; intros [Hvars Hconts]; (split; [ exact Hvars | ]);
    clear Hvars.
  - induction conts as [ | [k c] conts IH]; [ exact I | ].
    simpl in Hconts |- *. destruct Hconts as [Hc Hrest].
    split; [ exact Hc | exact (IH Hrest) ].
  - induction conts as [ | [k c] conts IH]; [ exact I | ].
    simpl in Hconts |- *. destruct Hconts as [Hc Hrest].
    split; [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma conts_closed_forall (h : heap)
    (conts : list (continuation * cont_value)) :
  conts_closed h conts
  <-> ∀ (k : continuation) (c : cont_value),
        (k, c) ∈ conts -> cont_value_closed h c.
Proof.
  induction conts as [ | [k c] conts [IH1 IH2]]; simpl.
  - split; [ intros _ k c [ ] | auto ].
  - split.
    + intros [Hc Hrest] k0 c0 [[= <- <-] | Hin];
        [ exact Hc | exact (IH1 Hrest _ _ Hin) ].
    + intros Hall. split; [ apply (Hall k c); left; reflexivity | ].
      apply IH2. intros k0 c0 Hin. apply (Hall k0 c0). right. exact Hin.
Qed.

Lemma cont_value_closed_evolves (h h1 : heap) (c : cont_value) :
  heap_evolves h h1 -> cont_value_closed h c -> cont_value_closed h1 c.
Proof.
  intros Hevolves.
  induction c
    as [handler vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_induction; [ | auto | | ].
  3: { simpl. intros (Hremaining & Hreturn & Hexn).
       split; [ | split; [ exact (IHreturn Hreturn) | exact (IHexn Hexn) ] ].
       intros v Hv.
       exact (value_closed_evolves _ _ _ Hevolves (Hremaining v Hv)). }
  2: { simpl. intros [Hextra Hc]. split; [ | exact (IH Hc) ].
       intros v Hv. exact (value_closed_evolves _ _ _ Hevolves (Hextra v Hv)). }
  intros Hc. apply cont_value_closed_closure in Hc as [Hvars Hconts].
  apply cont_value_closed_closure. split.
  - intros x v Hin. exact (value_closed_evolves _ _ _ Hevolves (Hvars x v Hin)).
  - apply conts_closed_forall. intros k c Hin.
    rewrite Forall_forall in IH. apply (IH (k, c) Hin).
    exact (proj1 (conts_closed_forall h conts) Hconts k c Hin).
Qed.

(* Evaluation in a closed environment gives closed values. *)
Lemma eval_simple_closed (h : heap) (vars : env) (s : simple) (v : value) :
  (∀ (x : variable) (w : value), (x, w) ∈ vars -> value_closed h w) ->
  eval_simple vars s = Some v ->
  value_closed h v.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - intros Hx. exact (Hvars x v (lookup_in _ _ _ Hx)).
  - intros [= <-] l [ ].
  - intros [= <-] l [ ].
Qed.

Lemma eval_simples_closed (h : heap) (vars : env) (args : list simple)
    (vs : list value) :
  (∀ (x : variable) (w : value), (x, w) ∈ vars -> value_closed h w) ->
  eval_simples vars args = Some vs ->
  ∀ v : value, v ∈ vs -> value_closed h v.
Proof.
  intros Hvars. revert vs. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-] v [ ].
  - destruct (eval_simple vars s) as [w | ] eqn:Hw; simpl; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; simpl;
      [ | discriminate ].
    intros [= <-] v [<- | Hin];
      [ exact (eval_simple_closed _ _ _ _ Hvars Hw)
      | exact (IH _ eq_refl v Hin) ].
Qed.

Lemma eval_value_slots_closed (h : heap) (vars : env)
    (slots : list (value_slot * simple)) (vs : list (value_slot * value)) :
  (∀ (x : variable) (w : value), (x, w) ∈ vars -> value_closed h w) ->
  eval_value_slots vars slots = Some vs ->
  ∀ v : value, v ∈ map snd vs -> value_closed h v.
Proof.
  intros Hvars. revert vs.
  induction slots as [ | [slot s] slots IH]; intros vs; simpl.
  - intros [= <-] v [ ].
  - destruct (eval_simple vars s) as [w | ] eqn:Hw; simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [ws | ] eqn:Hws; simpl;
      [ | discriminate ].
    intros [= <-] v [<- | Hin];
      [ exact (eval_simple_closed _ _ _ _ Hvars Hw)
      | exact (IH _ eq_refl v Hin) ].
Qed.

(* The fields of a closed tuple are closed. *)
Lemma tuple_fields_closed (st : static_env) (h : heap) :
  memory_closed st h ->
  FlambdaSemanticsProofs.P_fields (value_closed h) st.
Proof.
  intros Hmemory size tuple fields Htuple Hfields.
  apply (FlambdaSemanticsProofs.tuple_fields_forall _ st size tuple fields);
    [ | exact Hfields ].
  intros field v Hv.
  refine (eval_prim_closed st h _ [tuple] v Hmemory _ Hv).
  intros w [<- | [ ]]. exact Htuple.
Qed.

(* The closure at another slot of the set of a closed closure is
   closed. *)
Lemma project_function_slot_closed (st : static_env) (h : heap)
    (move_from move_to : function_slot) (v w : value) :
  value_closed h v ->
  project_function_slot st move_from move_to v = Some w ->
  value_closed h w.
Proof.
  intros Hclosed Hw.
  destruct v as [c | code vs | slot decls' vs | s | tag fields
    | elements | c | l | callee0 applied applied_arity];
    try discriminate.
  - cbn -[Pos.eqb] in Hw.
    destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    injection Hw as <-. exact Hclosed.
  - cbn -[Pos.eqb] in Hw.
    destruct (Pos.eqb slot move_from); [ | discriminate ].
    destruct (lookup slot decls'); simpl in Hw; [ | discriminate ].
    destruct (lookup move_to decls'); simpl in Hw; [ | discriminate ].
    injection Hw as <-. exact Hclosed.
  - destruct (project_symbol _ _ _ _ _ Hw) as (t & ->). intros l [ ].
Qed.

Lemma heap_exec_named_closed (st : static_env) (h h1 : heap) (vars : env)
    (d : named) (result : value) :
  memory_closed st h ->
  (∀ (x : variable) (w : value), (x, w) ∈ vars -> value_closed h w) ->
  heap_exec_named st h vars d = Some (result, h1) ->
  memory_closed st h1 ∧ value_closed h1 result.
Proof.
  intros Hmemory Hvars.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | project_from slot closure | move_from move_to closure];
    simpl.
  - destruct (eval_simple vars s) as [v | ] eqn:Hv; simpl; [ | discriminate ].
    intros [= <- <-].
    split; [ exact Hmemory | exact (eval_simple_closed _ _ _ _ Hvars Hv) ].
  - destruct (eval_simples vars args) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    apply exec_prim_closed;
      [ exact Hmemory | exact (eval_simples_closed _ _ _ _ Hvars Hvs) ].
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <- <-]. split; [ exact Hmemory | ].
    intros l Hl. simpl in Hl. apply in_flat_map in Hl as (slot & Hslot & Hl).
    exact
      (eval_value_slots_closed _ _ _ _ Hvars Hvs (snd slot)
         (in_map snd _ _ Hslot) l Hl).
  - destruct (lookup fslot decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <- <-]. split; [ exact Hmemory | ].
    intros l Hl. simpl in Hl. apply in_flat_map in Hl as (slot & Hslot & Hl).
    exact
      (eval_value_slots_closed _ _ _ _ Hvars Hvs (snd slot)
         (in_map snd _ _ Hslot) l Hl).
  - destruct (eval_simple vars closure) as [v | ] eqn:Hv; simpl;
      [ | discriminate ].
    destruct (closure_of st v) as [[code slots] | ] eqn:Hclosure; simpl;
      [ | discriminate ].
    destruct (lookup slot slots) as [w | ] eqn:Hw; simpl; [ | discriminate ].
    intros [= <- <-]. split; [ exact Hmemory | ].
    apply (closure_slots_closed st h v code slots);
      [ exact Hmemory
      | exact (eval_simple_closed _ _ _ _ Hvars Hv)
      | exact Hclosure
      | ].
    apply in_map_iff. exists (slot, w).
    split; [ reflexivity | exact (lookup_in _ _ _ Hw) ].
  - destruct (eval_simple vars closure) as [v | ] eqn:Hv; simpl;
      [ | discriminate ].
    unfold project_value_slot.
    destruct (closure_in_set st v) as [[[at_slot decls] slots] | ]
      eqn:Hclosure; simpl; [ | discriminate ].
    destruct (closure_in_set_closure_of _ _ _ _ _ Hclosure)
      as (code & _ & Hclosure_of).
    destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    destruct (lookup slot slots) as [w | ] eqn:Hw; simpl; [ | discriminate ].
    intros [= <- <-]. split; [ exact Hmemory | ].
    apply (closure_slots_closed st h v code slots);
      [ exact Hmemory
      | exact (eval_simple_closed _ _ _ _ Hvars Hv)
      | exact Hclosure_of
      | ].
    apply in_map_iff. exists (slot, w).
    split; [ reflexivity | exact (lookup_in _ _ _ Hw) ].
  - destruct (eval_simple vars closure) as [v | ] eqn:Hv; simpl;
      [ | discriminate ].
    destruct (project_function_slot st move_from move_to v) as [w | ]
      eqn:Hw; simpl; [ | discriminate ].
    intros [= <- <-]. split; [ exact Hmemory | ].
    exact
      (project_function_slot_closed _ _ _ _ _ _
         (eval_simple_closed _ _ _ _ Hvars Hv) Hw).
Qed.

Lemma eval_static_const_closed (h : heap) (vars : env) (const : static_const)
    (a : statically_allocated_value) :
  (∀ (x : variable) (w : value), (x, w) ∈ vars -> value_closed h w) ->
  eval_static_const vars const = Some a ->
  ∀ v : value, v ∈ values_of_static a -> value_closed h v.
Proof.
  intros Hvars.
  destruct const
    as [code slots | slot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-] v Hv. exact (eval_value_slots_closed _ _ _ _ Hvars Hvs v Hv).
  - destruct (lookup slot decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-] v Hv. exact (eval_value_slots_closed _ _ _ _ Hvars Hvs v Hv).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-] v Hv. exact (eval_simples_closed _ _ _ _ Hvars Hvs v Hv).
  - destruct (eval_simple vars contents) as [w | ] eqn:Hw; simpl;
      [ | discriminate ].
    intros [= <-] v [<- | [ ]]. exact (eval_simple_closed _ _ _ _ Hvars Hw).
  - intros [= <-] v [ ].
  - destruct (eval_simples vars elements) as [vs | ] eqn:Hvs; simpl;
      [ | discriminate ].
    intros [= <-] v Hv. exact (eval_simples_closed _ _ _ _ Hvars Hvs v Hv).
Qed.

Lemma eval_static_bindings_closed (h : heap) (vars : env)
    (bindings : list static_binding) :
  ∀ st st1 : static_env,
    (∀ (x : variable) (w : value), (x, w) ∈ vars -> value_closed h w) ->
    (∀ (s : symbol) (a : statically_allocated_value) (v : value),
      lookup s (static_values st) = Some a ->
      v ∈ values_of_static a ->
      value_closed h v) ->
    eval_static_bindings vars bindings st = Some st1 ->
    ∀ (s : symbol) (a : statically_allocated_value) (v : value),
      lookup s (static_values st1) = Some a ->
      v ∈ values_of_static a ->
      value_closed h v.
Proof.
  induction bindings as [ | binding bindings IH];
    intros st st1 Hvars Hstatics; simpl.
  - intros [= <-]. exact Hstatics.
  - destruct binding as [id c | id | s const].
    + destruct (lookup id (static_code st)); [ discriminate | ].
      apply IH; [ exact Hvars | exact Hstatics ].
    + apply IH; assumption.
    + destruct (lookup s (static_values st)); [ discriminate | ].
      destruct (eval_static_const vars const) as [a | ] eqn:Ha; simpl;
        [ | discriminate ].
      apply IH; [ exact Hvars | ].
      intros s0 a0 v. simpl. destruct (Pos.eqb s0 s).
      * intros [= <-]. exact (eval_static_const_closed _ _ _ _ Hvars Ha v).
      * apply Hstatics.
Qed.

Lemma bind_params_closed (h : heap) (params : list variable)
    (vs : list value) (vars env1 : env) :
  (∀ v : value, v ∈ vs -> value_closed h v) ->
  (∀ (x : variable) (w : value), (x, w) ∈ vars -> value_closed h w) ->
  bind_params params vs vars = Some env1 ->
  ∀ (x : variable) (w : value), (x, w) ∈ env1 -> value_closed h w.
Proof.
  revert vs vars. induction params as [ | x params IH];
    intros [ | v vs] vars Hvs Hvars; simpl; try discriminate.
  - intros [= <-]. exact Hvars.
  - apply IH.
    + intros w Hw. apply Hvs. right. exact Hw.
    + intros y w [[= <- <-] | Hin];
        [ apply Hvs; left; reflexivity | exact (Hvars y w Hin) ].
Qed.

Lemma resolve_closed (h : heap) (k : continuation) (conts : cont_env) :
  (∀ (k0 : continuation) (c : cont_value),
    (k0, c) ∈ conts -> cont_value_closed h c) ->
  cont_value_closed h (resolve k conts).
Proof.
  intros Hconts. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hc;
    [ exact (Hconts k c (lookup_in _ _ _ Hc)) | exact I ].
Qed.

Lemma rec_closures_closed (h : heap)
    (recursive : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  (∀ (x : variable) (w : value), (x, w) ∈ vars -> value_closed h w) ->
  (∀ (k : continuation) (c : cont_value),
    (k, c) ∈ conts -> cont_value_closed h c) ->
  ∀ (k : continuation) (c : cont_value),
    (k, c) ∈ rec_closures recursive vars conts ++ conts ->
    cont_value_closed h c.
Proof.
  intros Hvars Hconts k c Hin.
  apply in_app_or in Hin as [Hin | Hin]; [ | exact (Hconts k c Hin) ].
  unfold rec_closures in Hin.
  apply in_map_iff in Hin as ([k0 h0] & Heq & _). simpl in Heq.
  injection Heq as <- <-. apply cont_value_closed_closure.
  split; [ exact Hvars | apply conts_closed_forall; exact Hconts ].
Qed.

Lemma with_extra_args_closed (h : heap) (c : cont_value)
    (extra : list value) :
  cont_value_closed h c ->
  (∀ v : value, v ∈ extra -> value_closed h v) ->
  cont_value_closed h (with_extra_args c extra).
Proof.
  intros Hc Hextra. destruct extra as [ | v extra]; [ exact Hc | ].
  simpl. split; [ exact Hextra | exact Hc ].
Qed.

(* A partial application is closed when its callee and its arguments
   are. *)
Lemma value_closed_partial_application (h : heap) (callee : value)
    (args : list value) (args_arity : arity) :
  value_closed h (Value_partial_application callee args args_arity)
  <-> value_closed h callee ∧ Forall (value_closed h) args.
Proof.
  unfold value_closed. simpl. split.
  - intros Hclosed. split.
    + intros l Hl. apply Hclosed. apply in_or_app. left. exact Hl.
    + apply Forall_forall. intros arg Harg l Hl. apply Hclosed.
      apply in_or_app. right. apply in_flat_map. exists arg. auto.
  - intros [Hcallee Hargs] l Hl. apply in_app_or in Hl as [Hl | Hl].
    + exact (Hcallee l Hl).
    + apply in_flat_map in Hl as (arg & Harg & Hl).
      rewrite Forall_forall in Hargs. exact (Hargs arg Harg l Hl).
Qed.

(* A full call of a closed closure with closed arguments, returning to
   closed continuations. *)
Lemma heap_call_code_closed (st : static_env) (h : heap) (closure : value)
    (c : code) (vs : list value) (args_arity return_arity : arity)
    (ret exn : cont_value) (s' : heap_state) :
  memory_closed st h ->
  value_closed h closure ->
  (∀ v : value, v ∈ vs -> value_closed h v) ->
  cont_value_closed h ret ->
  cont_value_closed h exn ->
  heap_call_code closure c vs args_arity return_arity ret exn st h
  = Heap_step s' ->
  state_closed s'.
Proof.
  intros Hmemory Hclosure Hvs Hret Hexn. unfold heap_call_code.
  destruct (binds_statics (code_body c)); [ discriminate | ].
  destruct (negb _); [ discriminate | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [env1 | ] eqn:Hbind; [ | discriminate ].
  intros [= <-].
  split; [ exact Hmemory | split ]; cbn [hs_vars hs_conts hs_heap].
  - refine (bind_params_closed h _ vs _ env1 Hvs _ Hbind).
    intros y w [[= <- <-] | [ ]]. exact Hclosure.
  - intros k0 c0 [[= <- <-] | [[= <- <-] | [ ]]]; assumption.
Qed.

(* A call whose shape holds closed values. *)
Lemma heap_run_call_closed (st : static_env) (h : heap)
    (return_partial : value -> heap_step_result) (shape : option call_shape)
    (return_arity : arity) (ret exn : cont_value) (s' : heap_state) :
  memory_closed st h ->
  (∀ (partial : value) (s' : heap_state),
    value_closed h partial ->
    return_partial partial = Heap_step s' ->
    state_closed s') ->
  FlambdaSemanticsProofs.shape_values_all (value_closed h) shape ->
  cont_value_closed h ret ->
  cont_value_closed h exn ->
  heap_run_call return_partial shape return_arity ret exn st h
  = Heap_step s' ->
  state_closed s'.
Proof.
  intros Hmemory Hpartial Hshape Hret Hexn.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hshape |- *.
  - destruct Hshape as [Hclosure Hvs]. rewrite Forall_forall in Hvs.
    apply (heap_call_code_closed st h); assumption.
  - destruct Hshape as (Hclosure & Hvs & Hremaining).
    rewrite Forall_forall in Hvs, Hremaining.
    apply (heap_call_code_closed st h); try assumption.
    simpl. split; [ exact Hremaining | split; assumption ].
  - apply Hpartial. exact Hshape.
  - discriminate.
Qed.

(* Applying a closed continuation to closed values. *)
Lemma heap_apply_cont_value_closed (st : static_env) (h : heap)
    (c : cont_value) (vs : list value) (s' : heap_state) :
  memory_closed st h ->
  cont_value_closed h c ->
  (∀ v : value, v ∈ vs -> value_closed h v) ->
  heap_apply_cont_value c vs st h = Heap_step s' ->
  state_closed s'.
Proof.
  intros Hmemory. revert vs s'.
  induction c
    as [handler def_vars def_conts recursive _ | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_induction; intros vs s' Hc Hvs.
  4: {
    simpl in Hc. destruct Hc as (Hremaining & Hreturn & Hexn).
    cbn [heap_apply_cont_value].
    destruct vs as [ | v [ | v2 vs]]; try discriminate.
    apply (heap_run_call_closed st h); try assumption.
    - intros partial s0 Hpartial. apply IHreturn; [ exact Hreturn | ].
      intros w [<- | [ ]]. exact Hpartial.
    - apply FlambdaSemanticsProofs.shape_of_call_values_all.
      + exact (value_closed_partial_application h).
      + exact (tuple_fields_closed st h Hmemory).
      + apply Hvs. left. reflexivity.
      + apply Forall_forall. exact Hremaining. }
  - destruct handler as [params metadata handler]. simpl.
    apply cont_value_closed_closure in Hc as [Hdef_vars Hdef_conts].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars) as [env1 | ] eqn:Hbind;
      [ | discriminate ].
    intros [= <-].
    split; [ exact Hmemory | split ]; cbn [hs_vars hs_conts hs_heap].
    + exact (bind_params_closed h _ _ _ _ Hvs Hdef_vars Hbind).
    + apply rec_closures_closed; [ exact Hdef_vars | ].
      exact (proj1 (conts_closed_forall h def_conts) Hdef_conts).
  - discriminate.
  - simpl in Hc |- *. destruct Hc as [Hextra Hc]. apply IH; [ exact Hc | ].
    intros v Hv.
    apply in_app_or in Hv as [Hv | Hv];
      [ exact (Hvs v Hv) | exact (Hextra v Hv) ].
Qed.

Theorem heap_step_closed : heap_step_closed_statement.
Proof.
  intros [e vars conts st h] s' (Hmemory & Hvars & Hconts).
  cbn [hs_expr hs_vars hs_conts hs_statics hs_heap] in Hmemory, Hvars, Hconts.
  destruct e
    as [x x_kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ]; simpl.
  - (* Let *)
    destruct (heap_exec_named st h vars d) as [[v h1] | ] eqn:Hexec;
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) x_kind); [ | discriminate ].
    intros [= <-].
    destruct (heap_exec_named_closed _ _ _ _ _ _ Hmemory Hvars Hexec)
      as [Hmemory1 Hv].
    pose proof (heap_exec_named_evolves _ _ _ _ _ _ Hexec) as Hevolves.
    split; [ exact Hmemory1 | split ]; cbn [hs_vars hs_conts hs_heap].
    + intros y w [[= <- <-] | Hin];
        [ exact Hv
        | exact (value_closed_evolves _ _ _ Hevolves (Hvars y w Hin)) ].
    + intros k c Hin.
      exact (cont_value_closed_evolves _ _ _ Hevolves (Hconts k c Hin)).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings st) as [st1 | ] eqn:Heval;
      [ | discriminate ].
    intros [= <-]. split; [ | split; assumption ].
    split; [ exact (proj1 Hmemory) | ].
    exact
      (eval_static_bindings_closed _ _ _ _ _ Hvars (proj2 Hmemory) Heval).
  - (* Let_cont_non_recursive *)
    intros [= <-]. split; [ exact Hmemory | split; [ exact Hvars | ] ].
    cbn [hs_conts hs_heap].
    intros k0 c [[= <- <-] | Hin]; [ | exact (Hconts k0 c Hin) ].
    apply cont_value_closed_closure.
    split; [ exact Hvars | apply conts_closed_forall; exact Hconts ].
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-]. split; [ exact Hmemory | split; [ exact Hvars | ] ].
    cbn [hs_conts hs_heap]. apply rec_closures_closed; assumption.
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra; [ | discriminate ].
    unfold heap_call_function.
    apply (heap_run_call_closed st h); try assumption.
    + intros partial s0 Hpartial.
      apply (heap_apply_cont_value_closed st h);
        [ exact Hmemory | apply resolve_closed; exact Hconts | ].
      intros w [<- | [ ]]. exact Hpartial.
    + apply FlambdaSemanticsProofs.shape_of_call_values_all.
      * exact (value_closed_partial_application h).
      * exact (tuple_fields_closed st h Hmemory).
      * exact (eval_simple_closed _ _ _ _ Hvars Hcallee).
      * apply Forall_forall. exact (eval_simples_closed _ _ _ _ Hvars Hvs).
    + apply resolve_closed. exact Hconts.
    + apply with_extra_args_closed;
        [ apply resolve_closed; exact Hconts
        | exact (eval_simples_closed _ _ _ _ Hvars Hextra) ].
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    apply (heap_apply_cont_value_closed st h);
      [ exact Hmemory | apply resolve_closed; exact Hconts
      | exact (eval_simples_closed _ _ _ _ Hvars Hvs) ].
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. split; [ exact Hmemory | split; assumption ].
  - (* Invalid *)
    discriminate.
Qed.

(* ======================================================================== *)
(* Sharing and exchanging bindings                                          *)
(* ======================================================================== *)

(* A binding that the arguments do not mention does not change them. *)
Lemma eval_simples_unused (x : variable) (v : value) (vars : env)
    (args : list simple) :
  var_in_simples x args = false ->
  eval_simples ((x, v) :: vars) args = eval_simples vars args.
Proof.
  induction args as [ | s args IH]; [ reflexivity | ].
  unfold var_in_simples. simpl. intros Hunused.
  apply orb_false_iff in Hunused as [Hs Hargs].
  rewrite (IH Hargs).
  destruct s as [z | sym | c].
  - simpl in Hs |- *. rewrite (Pos.eqb_sym z x), Hs. reflexivity.
  - reflexivity.
  - reflexivity.
Qed.

Lemma heap_exec_named_prim_unused (st : static_env) (h : heap) (x : variable)
    (v : value) (vars : env) (p : prim) (args : list simple) :
  var_in_simples x args = false ->
  heap_exec_named st h ((x, v) :: vars) (Prim p args)
  = heap_exec_named st h vars (Prim p args).
Proof.
  intros Hunused. cbn [heap_exec_named].
  rewrite (eval_simples_unused _ _ _ _ Hunused). reflexivity.
Qed.

(* CSE, with nothing between the two applications. *)
Theorem cse_adjacent : cse_adjacent_statement.
Proof.
  intros x y x_kind y_kind p args body Heffects Hcoeffects Hdistinct Hunused
    vars conts st h
    Hclosed.
  destruct (heap_exec_named st h vars (Prim p args)) as [[v h1] | ] eqn:Hexec.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hexec.
       reflexivity. }
  destruct (kind_eqb (kind_of_value v) x_kind) eqn:Hkind_x.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hexec, Hkind_x.
       reflexivity. }
  assert (Hsame : h1 = h).
  { cbn [heap_exec_named] in Hexec.
    destruct (eval_simples vars args) as [vs | ]; [ | discriminate ].
    change (exec_prim st h p vs = Some (v, h1)) in Hexec.
    destruct Heffects as [Heff | Heff].
    - exact (no_effects _ _ _ _ _ _ Heff Hexec).
    - destruct (only_generative_effects _ _ _ _ _ _ _ Heff Hexec)
        as [[_ Hh] | (Hnot & _)];
        [ exact Hh | exfalso; apply Hnot; reflexivity ]. }
  subst h1.
  pose (s1 :=
          Heap_state (Let y y_kind (Prim p args) body) ((x, v) :: vars) conts
            st h).
  pose (s2 := Heap_state body ((y, v) :: (x, v) :: vars) conts st h).
  assert (Hstep1 :
            heap_step
              (Heap_state
                 (Let x x_kind (Prim p args)
                    (Let y y_kind (Prim p args) body))
                 vars conts st h)
            = Heap_step s1).
  { rewrite heap_step_let, Hexec, Hkind_x. reflexivity. }
  apply (heap_refines_after_source_step _ s1 _ Hstep1).
  destruct (kind_eqb (kind_of_value v) y_kind) eqn:Hkind_y.
  2: { apply heap_refines_undefined. unfold s1.
       rewrite heap_step_let,
         (heap_exec_named_prim_unused _ _ _ _ _ _ _ Hunused), Hexec, Hkind_y.
       reflexivity. }
  assert (Hstep2 : heap_step s1 = Heap_step s2).
  { unfold s1.
    rewrite heap_step_let, (heap_exec_named_prim_unused _ _ _ _ _ _ _ Hunused),
      Hexec, Hkind_y.
    reflexivity. }
  apply (heap_refines_after_source_step _ s2 _ Hstep2).
  apply
    (heap_refines_after_target_step _ _
       (Heap_state (Let y y_kind (Simple (Var x)) body) ((x, v) :: vars) conts
          st h)).
  { rewrite heap_step_let, Hexec, Hkind_x. reflexivity. }
  apply (heap_refines_after_target_step _ _ s2).
  { rewrite heap_step_let. cbn [heap_exec_named eval_named eval_simple]. simpl.
    rewrite Pos.eqb_refl, Hkind_y. reflexivity. }
  apply heap_refines_reflexive.
  apply (heap_step_closed s1 s2); [ | exact Hstep2 ].
  exact (heap_step_closed _ s1 Hclosed Hstep1).
Qed.

(* Two bindings exchanged, the first with neither effects nor
   coeffects. *)
Theorem exchange_lets : exchange_lets_statement.
Proof.
  intros x y x_kind y_kind p q args_p args_q body Heffects Hcoeffects
    Hdistinct Hx Hy vars
    conts st h Hclosed.
  destruct (heap_exec_named st h vars (Prim p args_p)) as [[vp hp] | ] eqn:Hp.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hp.
       reflexivity. }
  destruct (kind_eqb (kind_of_value vp) x_kind) eqn:Hkind_x.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hp, Hkind_x.
       reflexivity. }
  assert (Hpure :
            ∃ vs : list value,
              eval_simples vars args_p = Some vs
              ∧ exec_prim st h p vs = Some (vp, hp)).
  { cbn [heap_exec_named] in Hp.
    destruct (eval_simples vars args_p) as [vs | ]; [ | discriminate ].
    exists vs. split; [ reflexivity | exact Hp ]. }
  destruct Hpure as (vs & Hvs & Hexec_p).
  assert (Hsame : hp = h) by exact (no_effects _ _ _ _ _ _ Heffects Hexec_p).
  subst hp.
  pose (s1 :=
          Heap_state (Let y y_kind (Prim q args_q) body) ((x, vp) :: vars)
            conts st h).
  assert (Hstep1 :
            heap_step
              (Heap_state
                 (Let x x_kind (Prim p args_p)
                    (Let y y_kind (Prim q args_q) body))
                 vars conts st h)
            = Heap_step s1).
  { rewrite heap_step_let, Hp, Hkind_x. reflexivity. }
  apply (heap_refines_after_source_step _ s1 _ Hstep1).
  destruct (heap_exec_named st h vars (Prim q args_q)) as [[vq h2] | ] eqn:Hq.
  2: { apply heap_refines_undefined. unfold s1.
       rewrite heap_step_let, (heap_exec_named_prim_unused _ _ _ _ _ _ _ Hx),
         Hq.
       reflexivity. }
  destruct (kind_eqb (kind_of_value vq) y_kind) eqn:Hkind_y.
  2: { apply heap_refines_undefined. unfold s1.
       rewrite heap_step_let, (heap_exec_named_prim_unused _ _ _ _ _ _ _ Hx),
         Hq, Hkind_y.
       reflexivity. }
  pose (s2 := Heap_state body ((y, vq) :: (x, vp) :: vars) conts st h2).
  assert (Hstep2 : heap_step s1 = Heap_step s2).
  { unfold s1.
    rewrite heap_step_let, (heap_exec_named_prim_unused _ _ _ _ _ _ _ Hx), Hq,
      Hkind_y.
    reflexivity. }
  apply (heap_refines_after_source_step _ s2 _ Hstep2).
  pose (t1 :=
          Heap_state (Let x x_kind (Prim p args_p) body) ((y, vq) :: vars)
            conts st h2).
  pose (t2 := Heap_state body ((x, vp) :: (y, vq) :: vars) conts st h2).
  assert (Hp2 :
            heap_exec_named st h2 ((y, vq) :: vars) (Prim p args_p)
            = Some (vp, h2)).
  { rewrite (heap_exec_named_prim_unused _ _ _ _ _ _ _ Hy).
    cbn [heap_exec_named].
    rewrite Hvs.
    exact
      (no_coeffects _ _ _ _ _ _ _ Heffects Hcoeffects
         (heap_exec_named_evolves _ _ _ _ _ _ Hq) Hexec_p). }
  apply (heap_refines_after_target_step _ _ t1).
  { rewrite heap_step_let, Hq, Hkind_y. reflexivity. }
  apply (heap_refines_after_target_step _ _ t2).
  { unfold t1. rewrite heap_step_let, Hp2, Hkind_x. reflexivity. }
  apply machine_respects_relation.
  assert (Hclosed2 : state_closed s2).
  { apply (heap_step_closed s1 s2); [ | exact Hstep2 ].
    exact (heap_step_closed _ s1 Hclosed Hstep1). }
  destruct Hclosed2 as (Hmemory & Hvars & Hconts).
  cbn [s2 hs_expr hs_vars hs_conts hs_statics hs_heap]
    in Hmemory, Hvars, Hconts.
  apply (states_related_intro (identity_correspondence h2)).
  - apply closed_memories_related; [ exact Hmemory | auto ].
  - reflexivity.
  - intros z w _ Hw. exists w. split.
    + simpl in Hw |- *.
      destruct (Pos.eqb_spec z y) as [Heq_y | Hzy],
        (Pos.eqb_spec z x) as [Heq_x | Hzx]; try exact Hw.
      exfalso. apply Hdistinct. congruence.
    + apply closed_value_related. exact (Hvars z w (lookup_in _ _ _ Hw)).
  - apply (closed_conts_related _ h2); [ | exact Hconts ].
    intros w. apply closed_value_related.
Qed.

(* ======================================================================== *)
(* Lifting                                                                  *)
(* ======================================================================== *)

Lemma cont_code_free_of_closure (D : dropped) (h : continuation_handler)
    (vars : env) (conts : list (continuation * cont_value))
    (recursive : list (continuation * continuation_handler)) :
  cont_code_free_of D (Closure h vars conts recursive)
  = handler_free_of D h && conts_code_free_of D conts
    && forallb (fun '(_, h) => handler_free_of D h) recursive.
Proof. reflexivity. Qed.

Lemma cont_free_of_closure (D : dropped) (h : continuation_handler)
    (vars : env) (conts : list (continuation * cont_value))
    (recursive : list (continuation * continuation_handler)) :
  cont_free_of D (Closure h vars conts recursive)
  = handler_free_of D h && env_free_of D vars
    && forallb (fun '(_, c) => cont_free_of D c) conts
    && forallb (fun '(_, h) => handler_free_of D h) recursive.
Proof. reflexivity. Qed.

Lemma resolve_code_free_of (D : dropped) (k : continuation)
    (conts : cont_env) :
  conts_code_free_of D conts = true ->
  cont_code_free_of D (resolve k conts) = true.
Proof.
  intros Hconts. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hlookup; [ | reflexivity ].
  unfold conts_code_free_of in Hconts. rewrite forallb_forall in Hconts.
  exact (Hconts (k, c) (lookup_in _ _ _ Hlookup)).
Qed.

Lemma rec_closures_code_free_of (D : dropped)
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  forallb (fun '(_, h) => handler_free_of D h) handlers = true ->
  conts_code_free_of D conts = true ->
  conts_code_free_of D (rec_closures handlers vars conts ++ conts) = true.
Proof.
  intros Hhandlers Hconts. unfold conts_code_free_of. rewrite forallb_app.
  apply andb_true_iff. split; [ | exact Hconts ].
  pose proof Hhandlers as Hall. rewrite forallb_forall in Hall.
  unfold rec_closures. apply forallb_forall. intros [k c] Hin.
  apply in_map_iff in Hin as ([k0 h0] & [= <- <-] & Hin).
  rewrite cont_code_free_of_closure, Hconts, Hhandlers.
  rewrite (Hall (k0, h0) Hin). reflexivity.
Qed.

Lemma with_extra_args_code_free_of (D : dropped) (c : cont_value)
    (extra : list value) :
  cont_code_free_of D (with_extra_args c extra) = cont_code_free_of D c.
Proof. destruct extra; reflexivity. Qed.

(* A full call of code that names no symbol left out, returning to
   continuations whose handlers name none. *)
Lemma heap_call_code_code_free_of (D : dropped) (st : static_env) (h : heap)
    (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (ret exn : cont_value)
    (s1 : heap_state) :
  (∀ (id : code_id) (c : code),
    lookup id (static_code st) = Some c -> code_free_of D c = true) ->
  code_free_of D c = true ->
  cont_code_free_of D ret = true ->
  cont_code_free_of D exn = true ->
  heap_call_code closure c vs args_arity return_arity ret exn st h
  = Heap_step s1 ->
  state_code_free_of D s1.
Proof.
  intros Hcode Hc Hret Hexn. unfold heap_call_code.
  destruct (binds_statics (code_body c)); [ discriminate | ].
  destruct (negb _); [ discriminate | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ]; [ | discriminate ].
  intros [= <-].
  unfold code_free_of in Hc.
  apply andb_true_iff in Hc as [Hc_avoids Hc_binds].
  destruct c as [params my_closure ret0 exn0 metadata code_body].
  simpl in Hc_avoids, Hc_binds.
  split; [ exact Hc_avoids | ]. split; [ exact Hc_binds | ].
  split; [ | exact Hcode ].
  cbn [hs_conts]. unfold conts_code_free_of. cbn [forallb].
  rewrite Hret, Hexn. reflexivity.
Qed.

(* A call of code that is bound. *)
Lemma heap_run_call_code_free_of (D : dropped) (st : static_env) (h : heap)
    (return_partial : value -> heap_step_result) (kind : call_kind)
    (shape : option call_shape) (return_arity : arity)
    (ret exn : cont_value) (s1 : heap_state) :
  (∀ (id : code_id) (c : code),
    lookup id (static_code st) = Some c -> code_free_of D c = true) ->
  (∀ (partial : value) (s1 : heap_state),
    return_partial partial = Heap_step s1 -> state_code_free_of D s1) ->
  FlambdaSemanticsProofs.shape_calls st kind shape ->
  cont_code_free_of D ret = true ->
  cont_code_free_of D exn = true ->
  heap_run_call return_partial shape return_arity ret exn st h
  = Heap_step s1 ->
  state_code_free_of D s1.
Proof.
  intros Hcode Hpartial Hcalls Hret Hexn.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    apply (heap_call_code_code_free_of D st h); try assumption.
    exact (Hcode called c Hlookup).
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    apply (heap_call_code_code_free_of D st h); try assumption.
    + exact (Hcode called c Hlookup).
    + simpl. rewrite Hret, Hexn. reflexivity.
  - apply Hpartial.
  - discriminate.
Qed.

(* Applying a continuation whose handlers name no symbol left out. *)
Lemma heap_apply_cont_value_code_free_of (D : dropped) (st : static_env)
    (h : heap) (c : cont_value) (vs : list value) (s1 : heap_state) :
  (∀ (id : code_id) (c : code),
    lookup id (static_code st) = Some c -> code_free_of D c = true) ->
  cont_code_free_of D c = true ->
  heap_apply_cont_value c vs st h = Heap_step s1 ->
  state_code_free_of D s1.
Proof.
  intros Hcode. revert vs s1.
  induction c
    as [handler def_vars def_conts recursive _ | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_induction; intros vs s1 Hc.
  4: {
    simpl in Hc. apply andb_true_iff in Hc as [Hreturn Hexn].
    cbn [heap_apply_cont_value].
    destruct vs as [ | v [ | v2 vs]]; try discriminate.
    apply (heap_run_call_code_free_of D st h _ Indirect); try assumption.
    - intros partial s0. apply IHreturn. exact Hreturn.
    - apply FlambdaSemanticsProofs.shape_of_call_calls. }
  all: simpl.
  - destruct handler as [params metadata handler].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      [ | discriminate ].
    intros [= <-]. rewrite cont_code_free_of_closure in Hc.
    apply andb_true_iff in Hc as [Hc Hrec].
    apply andb_true_iff in Hc as [Hhandler Hdef].
    unfold handler_free_of in Hhandler.
    apply andb_true_iff in Hhandler as [Hh_avoids Hh_binds].
    simpl in Hh_avoids, Hh_binds.
    split; [ exact Hh_avoids | ]. split; [ exact Hh_binds | ].
    split; [ | exact Hcode ].
    exact (rec_closures_code_free_of D recursive def_vars def_conts Hrec Hdef).
  - discriminate.
  - apply IH. exact Hc.
Qed.

Lemma eval_static_bindings_code_free_of (D : dropped) (vars : env)
    (bindings : list static_binding) :
  ∀ st st1 : static_env,
    forallb (binding_avoids D) bindings = true ->
    forallb (binding_binds_only_kept D) bindings = true ->
    (∀ (id : code_id) (c : code),
      lookup id (static_code st) = Some c -> code_free_of D c = true) ->
    eval_static_bindings vars bindings st = Some st1 ->
    ∀ (id : code_id) (c : code),
      lookup id (static_code st1) = Some c -> code_free_of D c = true.
Proof.
  induction bindings as [ | binding bindings IH];
    intros st st1 Havoid Hbinds Hcode; simpl.
  - intros [= <-]. exact Hcode.
  - simpl in Havoid, Hbinds.
    apply andb_true_iff in Havoid as [Havoid_head Havoid_tail].
    apply andb_true_iff in Hbinds as [Hbinds_head Hbinds_tail].
    destruct binding as [id0 c0 | id0 | s const].
    + destruct (lookup id0 (static_code st)); [ discriminate | ].
      apply (IH _ _ Havoid_tail Hbinds_tail).
      intros id c. simpl. destruct (Pos.eqb id id0); [ | apply Hcode ].
      intros [= <-]. simpl in Havoid_head, Hbinds_head.
      apply andb_true_iff in Hbinds_head as [Hkept Hbinds_code].
      rewrite Hkept in Havoid_head. unfold code_free_of.
      rewrite Havoid_head, Hbinds_code. reflexivity.
    + exact (IH _ _ Havoid_tail Hbinds_tail Hcode).
    + destruct (lookup s (static_values st)); [ discriminate | ].
      destruct (eval_static_const vars const) as [a | ]; simpl;
        [ | discriminate ].
      apply (IH _ _ Havoid_tail Hbinds_tail). exact Hcode.
Qed.

(* A step keeps the invariant. *)
Theorem heap_step_code_free_of : heap_step_code_free_of_statement.
Proof.
  intros D s s1. destruct s as [e vars conts st h].
  intros (Havoid & Hbinds & Hconts & Hcode).
  cbn [hs_expr hs_conts hs_statics] in Havoid, Hbinds, Hconts, Hcode.
  destruct e
    as [x x_kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ];
    simpl in Havoid, Hbinds |- *.
  - (* Let *)
    destruct (heap_exec_named st h vars d) as [[v h1] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) x_kind); [ | discriminate ].
    intros [= <-]. apply andb_true_iff in Havoid as [_ Havoid].
    split; [ exact Havoid | ]. split; [ exact Hbinds | ].
    split; [ exact Hconts | exact Hcode ].
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings st) as [st1 | ] eqn:Heval;
      [ | discriminate ].
    intros [= <-].
    apply andb_true_iff in Havoid as [Havoid_bindings Havoid].
    apply andb_true_iff in Hbinds as [Hbinds_bindings Hbinds].
    split; [ exact Havoid | ]. split; [ exact Hbinds | ].
    split; [ exact Hconts | ].
    exact
      (eval_static_bindings_code_free_of D vars bindings st st1
         Havoid_bindings Hbinds_bindings Hcode Heval).
  - (* Let_cont_non_recursive *)
    intros [= <-].
    apply andb_true_iff in Havoid as [Havoid_handler Havoid].
    apply andb_true_iff in Hbinds as [Hbinds_handler Hbinds].
    split; [ exact Havoid | ]. split; [ exact Hbinds | ].
    split; [ | exact Hcode ].
    change
      (cont_code_free_of D (Closure handler vars conts [])
       && conts_code_free_of D conts
       = true).
    rewrite cont_code_free_of_closure. unfold handler_free_of.
    rewrite Havoid_handler, Hbinds_handler, Hconts. reflexivity.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-].
    apply andb_true_iff in Havoid as [Havoid_handlers Havoid].
    apply andb_true_iff in Hbinds as [Hbinds_handlers Hbinds].
    split; [ exact Havoid | ]. split; [ exact Hbinds | ].
    split; [ | exact Hcode ].
    apply rec_closures_code_free_of; [ | exact Hconts ].
    apply forallb_forall. intros [k0 h0] Hin.
    rewrite forallb_forall in Havoid_handlers, Hbinds_handlers.
    specialize (Havoid_handlers _ Hin). specialize (Hbinds_handlers _ Hin).
    simpl in Havoid_handlers, Hbinds_handlers. unfold handler_free_of.
    rewrite Havoid_handlers, Hbinds_handlers. reflexivity.
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    unfold heap_call_function.
    apply (heap_run_call_code_free_of D st h _ (ap_call_kind ap));
      try assumption.
    + intros partial s0.
      apply (heap_apply_cont_value_code_free_of D st h); [ exact Hcode | ].
      exact (resolve_code_free_of D _ conts Hconts).
    + apply FlambdaSemanticsProofs.shape_of_call_calls.
    + exact (resolve_code_free_of D _ conts Hconts).
    + rewrite with_extra_args_code_free_of.
      exact (resolve_code_free_of D _ conts Hconts).
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    apply (heap_apply_cont_value_code_free_of D st h); [ exact Hcode | ].
    exact (resolve_code_free_of D (ac_continuation ac) conts Hconts).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | discriminate ].
    intros [= <-]. apply andb_true_iff in Havoid as [_ Harms].
    destruct (find_arm_in _ _ _ Harm) as (d0 & Hin).
    rewrite forallb_forall in Harms.
    split; [ exact (Harms (d0, ac) Hin) | ]. split; [ reflexivity | ].
    split; [ exact Hconts | exact Hcode ].
  - (* Invalid *)
    discriminate.
Qed.

(* Under the invariant, the next step names no symbol left out. *)
Lemma state_code_free_of_head (D : dropped) (s : heap_state) :
  state_code_free_of D s -> head_avoids D (hs_expr s) = true.
Proof.
  destruct s as [e vars conts st h]. intros (Havoid & Hbinds & _).
  cbn [hs_expr] in Havoid, Hbinds |- *.
  destruct e
    as [x x_kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ];
    simpl in Havoid, Hbinds |- *; try reflexivity.
  - apply andb_true_iff in Havoid as [Hd _]. exact Hd.
  - apply andb_true_iff in Havoid as [Havoid _].
    apply andb_true_iff in Hbinds as [Hbinds _].
    apply forallb_forall. intros binding Hin.
    rewrite forallb_forall in Havoid, Hbinds.
    specialize (Havoid _ Hin). specialize (Hbinds _ Hin).
    destruct binding as [id c | id | sym const];
      simpl in Havoid, Hbinds |- *; try reflexivity.
    rewrite Hbinds in Havoid |- *. exact Havoid.
  - exact Havoid.
  - exact Havoid.
  - apply andb_true_iff in Havoid as [Hscrutinee _]. exact Hscrutinee.
Qed.

Theorem machine_respects_relation_except :
  machine_respects_relation_except_statement.
Proof.
  intros D.
  exact
    (heap_refines_except D (state_code_free_of D) (heap_step_code_free_of D)
       (state_code_free_of_head D)).
Qed.

(* What FlambdaUnusedStaticsSpec.cont_free_of says of the handlers. *)
Lemma cont_free_of_code (D : dropped) (c : cont_value) :
  cont_free_of D c = true -> cont_code_free_of D c = true.
Proof.
  induction c
    as [handler vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_induction; [ | reflexivity | | ].
  3: { simpl. intros Hfree. apply andb_true_iff in Hfree as [Hfree Hexn].
       apply andb_true_iff in Hfree as [_ Hreturn].
       rewrite (IHreturn Hreturn), (IHexn Hexn). reflexivity. }
  2: { simpl. intros Hfree. apply andb_true_iff in Hfree as [Hfree _].
       exact (IH Hfree). }
  intros Hfree. rewrite cont_free_of_closure in Hfree.
  apply andb_true_iff in Hfree as [Hfree Hrec].
  apply andb_true_iff in Hfree as [Hfree Hconts].
  apply andb_true_iff in Hfree as [Hhandler _].
  rewrite cont_code_free_of_closure, Hhandler, Hrec. simpl.
  rewrite andb_true_r.
  unfold conts_code_free_of. apply forallb_forall. intros [k c] Hin.
  rewrite Forall_forall in IH. rewrite forallb_forall in Hconts.
  exact (IH (k, c) Hin (Hconts (k, c) Hin)).
Qed.

(* A closed continuation whose values refer to no symbol left out is
   related to itself. *)
Lemma free_cont_related (D : dropped) (R : value -> value -> Prop) (h : heap)
    (c : cont_value) :
  (∀ v : value, value_closed h v -> value_avoids D v = true -> R v v) ->
  cont_value_closed h c ->
  cont_free_of D c = true ->
  cont_related R c c.
Proof.
  intros Hsame.
  induction c
    as [handler vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_induction; intros Hclosed Hfree.
  4: { simpl in Hclosed, Hfree.
       destruct Hclosed as (Hremaining & Hreturn & Hexn).
       apply andb_true_iff in Hfree as [Hfree Hexn_free].
       apply andb_true_iff in Hfree as [Hremaining_free Hreturn_free].
       apply Related_over_application;
         [ | exact (IHreturn Hreturn Hreturn_free)
         | exact (IHexn Hexn Hexn_free) ].
       apply Forall2_same. intros v Hv.
       apply Hsame; [ exact (Hremaining v Hv) | ].
       rewrite forallb_forall in Hremaining_free.
       exact (Hremaining_free v Hv). }
  3: { simpl in Hclosed, Hfree. destruct Hclosed as [Hextra Hc].
       apply andb_true_iff in Hfree as [Hc_free Hextra_free].
       apply Related_extra_args; [ exact (IH Hc Hc_free) | ].
       apply Forall2_same. intros v Hv.
       apply Hsame; [ exact (Hextra v Hv) | ].
       rewrite forallb_forall in Hextra_free. exact (Hextra_free v Hv). }
  - simpl in Hclosed. destruct Hclosed as [Hvars Hconts].
    rewrite cont_free_of_closure in Hfree.
    apply andb_true_iff in Hfree as [Hfree _].
    apply andb_true_iff in Hfree as [Hfree Hconts_free].
    apply andb_true_iff in Hfree as [_ Henv_free].
    apply Related_closure.
    + intros x v _ Hv. exists v. split; [ exact Hv | ].
      apply Hsame; [ exact (Hvars x v (lookup_in _ _ _ Hv)) | ].
      unfold env_free_of in Henv_free. rewrite forallb_forall in Henv_free.
      exact (Henv_free (x, v) (lookup_in _ _ _ Hv)).
    + revert IH Hconts Hconts_free.
      induction conts as [ | [k c] conts IHconts];
        intros IH Hconts Hconts_free; [ constructor | ].
      destruct Hconts as [Hc Hrest]. simpl in Hconts_free.
      apply andb_true_iff in Hconts_free as [Hc_free Hrest_free].
      inversion IH as [ | b l Hb Hl]; subst.
      constructor;
        [ split; [ reflexivity | exact (Hb Hc Hc_free) ]
        | exact (IHconts Hl Hrest Hrest_free) ].
  - apply Related_outer.
Qed.

(* The values of simples that a primitive accepts refer to no symbol
   left out, where the environment refers to none and no symbol left out
   is bound. *)
Lemma eval_simples_avoid_bound (D : dropped) (statics : static_env) (h : heap)
    (vars : env) (args : list simple) (vs : list value) :
  env_free_of D vars = true ->
  (∀ t : symbol,
    is_bound t (static_values statics) = true -> symbol_kept D t = true) ->
  eval_simples vars args = Some vs ->
  forallb (argument_bound statics h) vs = true ->
  ∀ v : value, v ∈ vs -> value_avoids D v = true.
Proof.
  intros Hvars Hkept. revert vs.
  induction args as [ | a args IH]; intros vs; simpl.
  - intros [= <-] _ v [ ].
  - destruct (eval_simple vars a) as [w | ] eqn:Hw; simpl; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; simpl;
      [ | discriminate ].
    intros [= <-]. simpl. intros Hbound.
    apply andb_true_iff in Hbound as [Hbound_w Hbound_ws].
    intros v [<- | Hin]; [ | exact (IH _ eq_refl Hbound_ws v Hin) ].
    destruct a as [y | t | c]; simpl in Hw.
    + unfold env_free_of in Hvars. rewrite forallb_forall in Hvars.
      exact (Hvars (y, w) (lookup_in _ _ _ Hw)).
    + injection Hw as <-. simpl in Hbound_w |- *. exact (Hkept t Hbound_w).
    + injection Hw as <-. reflexivity.
Qed.

(* What FlambdaUnusedStaticsSpec.allocated_free_of says of the values
   that a symbol holds, and of the symbols that a static set names. *)
Lemma allocated_free_of_values (D : dropped) (a : statically_allocated_value)
    (v : value) :
  allocated_free_of D a = true ->
  v ∈ values_of_static a ->
  value_avoids D v = true.
Proof.
  destruct a
    as [code slots | slot syms decls slots | tag fields | contents | bytes
        | elements | ];
    unfold allocated_free_of, values_of_static; intros Ha Hv.
  - apply andb_true_iff in Ha as [_ Ha].
    apply in_map_iff in Hv as ([name w] & <- & Hin).
    rewrite forallb_forall in Ha. exact (Ha (name, w) Hin).
  - apply andb_true_iff in Ha as [_ Ha].
    apply in_map_iff in Hv as ([name w] & <- & Hin).
    rewrite forallb_forall in Ha. exact (Ha (name, w) Hin).
  - rewrite forallb_forall in Ha. exact (Ha v Hv).
  - destruct Hv as [<- | [ ]]. exact Ha.
  - destruct Hv.
  - rewrite forallb_forall in Ha. exact (Ha v Hv).
  - destruct Hv.
Qed.

Lemma allocated_free_of_sets (D : dropped) (slot : function_slot)
    (syms : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  allocated_free_of D (Static_closure_of_set slot syms decls slots) = true ->
  closure_symbols_avoid D syms = true.
Proof.
  unfold allocated_free_of. intros Ha.
  apply andb_true_iff in Ha as [Ha _].
  apply andb_true_iff in Ha as [Ha _]. exact Ha.
Qed.

(* What lifting comes to. Two states run the same body; the second binds
   more symbols, to things that are allocated, and binds the variable to
   one of them, where the first binds it to a value that is related to
   the symbol. The two are related up to the new symbols, which the first
   neither binds nor names. *)
Lemma lifted_refines (D : dropped) (x : variable) (s : symbol) (v : value)
    (body : expr) (vars : env) (conts : cont_env)
    (statics statics1 : static_env) (h : heap) :
  static_code statics = static_code statics1 ->
  (∀ t : symbol,
    symbol_kept D t = true ->
    lookup t (static_values statics1) = lookup t (static_values statics)) ->
  (∀ t : symbol,
    symbol_kept D t = false ->
    lookup t (static_values statics) = None
    ∧ is_allocated statics1 (Value_symbol t) = true) ->
  expr_avoids D body = true ->
  binds_only_kept D body = true ->
  envs_free_of D vars conts statics = true ->
  heap_free_of D h = true ->
  state_closed (Heap_state body vars conts statics h) ->
  value_related false (identity_correspondence_except D h) statics statics1 v
    (Value_symbol s) ->
  heap_refines (Heap_state body ((x, v) :: vars) conts statics h)
    (Heap_state body ((x, Value_symbol s) :: vars) conts statics1 h).
Proof.
  intros Hcode Hagree Hleft_out Hbody_avoids Hbody_binds Henvs Hheap_free
    Hclosed Hlifted.
  destruct Hclosed as (Hmemory & Hvars & Hconts).
  cbn [hs_expr hs_vars hs_conts hs_statics hs_heap] in Hmemory, Hvars, Hconts.
  unfold envs_free_of in Henvs.
  apply andb_true_iff in Henvs as [Henvs Hstatics_free].
  apply andb_true_iff in Henvs as [Hvars_free Hconts_free].
  unfold statics_free_of in Hstatics_free.
  apply andb_true_iff in Hstatics_free as [Hcode_free Hvalues_free].
  rewrite forallb_forall in Hconts_free, Hcode_free, Hvalues_free.
  apply (machine_respects_relation_except D).
  - split; [ exact Hbody_avoids | ]. split; [ exact Hbody_binds | ]. split.
    + apply forallb_forall. intros [k c] Hin.
      exact (cont_free_of_code D c (Hconts_free (k, c) Hin)).
    + intros id c Hlookup.
      exact (Hcode_free (id, c) (lookup_in _ _ _ Hlookup)).
  - assert (Hstatics_avoid :
              ∀ (t : symbol) (a0 : statically_allocated_value) (w : value),
                lookup t (static_values statics) = Some a0 ->
                w ∈ values_of_static a0 ->
                value_avoids D w = true).
    { intros t a0 w Hlookup.
      exact
        (allocated_free_of_values D a0 w
           (Hvalues_free (t, a0) (lookup_in _ _ _ Hlookup))). }
    assert (Hsets : static_sets_kept D statics).
    { intros t slot syms decls slots Hlookup.
      exact
        (allocated_free_of_sets D _ _ _ _
           (Hvalues_free (t, _) (lookup_in _ _ _ Hlookup))). }
    assert (Hcells :
              ∀ (l : location) (c : cell) (w : value),
                lookup l h = Some c ->
                w ∈ values_of_cell c ->
                value_avoids D w = true).
    { intros l c w Hlookup Hw. unfold heap_free_of in Hheap_free.
      rewrite forallb_forall in Hheap_free.
      pose proof (Hheap_free (l, c) (lookup_in _ _ _ Hlookup)) as Hc.
      simpl in Hc. rewrite forallb_forall in Hc. exact (Hc w Hw). }
    assert (Hmemories :
              memories_related D (identity_correspondence_except D h) statics
                statics1 h h).
    { apply closed_memories_related_except; try assumption. auto. }
    assert (Hsame :
              ∀ w : value,
                value_closed h w ->
                value_avoids D w = true ->
                value_related false (identity_correspondence_except D h)
                  statics statics1 w w).
    { intros w. apply closed_value_related_except. }
    apply (states_related_except_intro D (identity_correspondence_except D h)).
    + exact Hmemories.
    + exact Hcode.
    + intros y w _. simpl. destruct (Pos.eqb y x).
      * intros [= <-]. exists (Value_symbol s).
        split; [ reflexivity | exact Hlifted ].
      * intros Hw. exists w. split; [ exact Hw | ]. apply Hsame.
        -- exact (Hvars y w (lookup_in _ _ _ Hw)).
        -- unfold env_free_of in Hvars_free.
           rewrite forallb_forall in Hvars_free.
           exact (Hvars_free (y, w) (lookup_in _ _ _ Hw)).
    + apply Forall2_same. intros [k c] Hin. split; [ reflexivity | ].
      apply (free_cont_related D _ h);
        [ exact Hsame | exact (Hconts k c Hin)
        | exact (Hconts_free (k, c) Hin) ].
Qed.

(* The same, where the second state binds one more symbol. *)
Lemma lifted_refines_one (x : variable) (s : symbol) (v : value)
    (a : statically_allocated_value) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (h : heap) :
  lookup s (static_values statics) = None ->
  expr_avoids (Dropped [s] []) body = true ->
  binds_only_kept (Dropped [s] []) body = true ->
  envs_free_of (Dropped [s] []) vars conts statics = true ->
  heap_free_of (Dropped [s] []) h = true ->
  state_closed (Heap_state body vars conts statics h) ->
  is_allocated (bind_symbol statics s a) (Value_symbol s) = true ->
  value_related false (identity_correspondence_except (Dropped [s] []) h)
    statics (bind_symbol statics s a) v (Value_symbol s) ->
  heap_refines (Heap_state body ((x, v) :: vars) conts statics h)
    (Heap_state body ((x, Value_symbol s) :: vars) conts
       (bind_symbol statics s a) h).
Proof.
  intros Hs Hbody_avoids Hbody_binds Henvs Hheap_free Hclosed Hallocated
    Hlifted.
  apply (lifted_refines (Dropped [s] [])); try assumption.
  - reflexivity.
  - intros t Hkept. simpl. unfold symbol_kept in Hkept. simpl in Hkept.
    destruct (Pos.eqb t s); [ discriminate | reflexivity ].
  - intros t Hdropped. unfold symbol_kept in Hdropped. simpl in Hdropped.
    destruct (Pos.eqb_spec t s) as [Heq | _]; [ | discriminate ].
    rewrite Heq. split; [ exact Hs | exact Hallocated ].
Qed.

(* A symbol that the static environment binds is not the new one. *)
Lemma bound_symbol_kept (s : symbol) (statics : static_env) :
  lookup s (static_values statics) = None ->
  ∀ t : symbol,
    is_bound t (static_values statics) = true ->
    symbol_kept (Dropped [s] []) t = true.
Proof.
  intros Hs t Ht. unfold symbol_kept. simpl.
  destruct (Pos.eqb_spec t s) as [-> | _]; [ | reflexivity ].
  unfold is_bound in Ht. rewrite Hs in Ht. discriminate.
Qed.

(* Lifting a block. The first state binds the variable to the block; the
   second binds the new symbol to it and the variable to the symbol. *)
Theorem lift_block : lift_block_statement.
Proof.
  intros x x_kind s tag fields body vars conts statics h Hs Hbody_avoids
    Hbody_binds
    Henvs Hheap_free Hclosed.
  destruct
    (heap_exec_named statics h vars
       (Prim (Variadic (Make_block (Block_kind_values tag))) fields))
    as [[v h1] | ] eqn:Hexec.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hexec.
       reflexivity. }
  pose proof Hexec as Hexec0. cbn [heap_exec_named] in Hexec.
  destruct (eval_simples vars fields) as [vs | ] eqn:Hvs; [ | discriminate ].
  change
    (exec_prim statics h (Variadic (Make_block (Block_kind_values tag))) vs
     = Some (v, h1))
    in Hexec.
  destruct (exec_prim_bound _ _ _ _ _ _ Hexec) as [Hbound Hexec_bound].
  unfold exec_bound_prim, eval_prim, eval_variadic_primitive in Hexec_bound.
  destruct (_ && _) in Hexec_bound; [ | discriminate ].
  simpl in Hexec_bound. injection Hexec_bound as <- <-.
  destruct (kind_eqb Value x_kind) eqn:Hkind.
  2: { apply heap_refines_undefined. rewrite heap_step_let, Hexec0.
       cbn [kind_of_value]. rewrite Hkind. reflexivity. }
  unfold kind_eqb in Hkind.
  destruct (kind_eq_dec Value x_kind) as [<- | _];
    [ clear Hkind | discriminate ].
  pose (x_kind := Value).
  apply
    (heap_refines_after_source_step _
       (Heap_state body ((x, Value_block tag vs) :: vars) conts statics h));
    [ rewrite heap_step_let, Hexec0; reflexivity | ].
  apply
    (heap_refines_after_target_step _ _
       (Heap_state (Let x x_kind (Simple (Symbol s)) body) vars conts
          (bind_symbol statics s (Static_block tag vs)) h)).
  { simpl. rewrite Hs, Hvs. reflexivity. }
  apply
    (heap_refines_after_target_step _ _
       (Heap_state body ((x, Value_symbol s) :: vars) conts
          (bind_symbol statics s (Static_block tag vs)) h));
    [ reflexivity | ].
  apply lifted_refines_one; try assumption.
  - unfold is_allocated. simpl. rewrite Pos.eqb_refl. reflexivity.
  - apply (related_block _ _ _ _ _ _ tag vs vs);
      [ reflexivity | simpl; rewrite Pos.eqb_refl; reflexivity | ].
    apply Forall2_same. intros w Hw. apply closed_value_related_except.
    + exact
        (eval_simples_closed h vars fields vs (proj1 (proj2 Hclosed)) Hvs w
           Hw).
    + unfold envs_free_of in Henvs.
      apply andb_true_iff in Henvs as [Henvs _].
      apply andb_true_iff in Henvs as [Hvars_free _].
      exact
        (eval_simples_avoid_bound _ statics h vars fields vs Hvars_free
           (bound_symbol_kept s statics Hs) Hvs Hbound w Hw).
Qed.

(* The values of value slots that name no symbol left out refer to none,
   where the environment refers to none. *)
Lemma eval_value_slots_avoid (D : dropped) (vars : env)
    (slots : list (value_slot * simple)) (vs : list (value_slot * value)) :
  env_free_of D vars = true ->
  value_slots_avoid D slots = true ->
  eval_value_slots vars slots = Some vs ->
  ∀ v : value, v ∈ map snd vs -> value_avoids D v = true.
Proof.
  intros Hvars. revert vs.
  induction slots as [ | [slot a] slots IH]; intros vs; simpl.
  - intros _ [= <-] v [ ].
  - intros Havoid. apply andb_true_iff in Havoid as [Havoid_a Havoid_slots].
    simpl in Havoid_a.
    destruct (eval_simple vars a) as [w | ] eqn:Hw; simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [ws | ] eqn:Hws; simpl;
      [ | discriminate ].
    intros [= <-] v [<- | Hin]; [ | exact (IH _ Havoid_slots eq_refl v Hin) ].
    destruct a as [y | t | c]; simpl in Hw.
    + unfold env_free_of in Hvars. rewrite forallb_forall in Hvars.
      exact (Hvars (y, w) (lookup_in _ _ _ Hw)).
    + injection Hw as <-. exact Havoid_a.
    + injection Hw as <-. reflexivity.
Qed.

(* Lifting the closure of one function, in the same way. *)
Theorem lift_closure : lift_closure_statement.
Proof.
  intros x x_kind s code value_slots body vars conts statics h Hs
    Hslots_avoid
    Hbody_avoids Hbody_binds Henvs Hheap_free Hclosed.
  destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs.
  2: { apply heap_refines_undefined. rewrite heap_step_let.
       cbn [heap_exec_named eval_named]. rewrite Hvs. reflexivity. }
  destruct (kind_eqb Value x_kind) eqn:Hkind.
  2: { apply heap_refines_undefined. rewrite heap_step_let.
       cbn [heap_exec_named eval_named]. rewrite Hvs. cbn [kind_of_value].
       rewrite Hkind. reflexivity. }
  unfold kind_eqb in Hkind.
  destruct (kind_eq_dec Value x_kind) as [<- | _];
    [ clear Hkind | discriminate ].
  pose (x_kind := Value).
  apply
    (heap_refines_after_source_step _
       (Heap_state body ((x, Value_closure code vs) :: vars) conts statics h)).
  { rewrite heap_step_let. cbn [heap_exec_named eval_named]. rewrite Hvs.
    reflexivity. }
  apply
    (heap_refines_after_target_step _ _
       (Heap_state (Let x x_kind (Simple (Symbol s)) body) vars conts
          (bind_symbol statics s (Static_closure code vs)) h)).
  { simpl. rewrite Hs, Hvs. reflexivity. }
  apply
    (heap_refines_after_target_step _ _
       (Heap_state body ((x, Value_symbol s) :: vars) conts
          (bind_symbol statics s (Static_closure code vs)) h));
    [ reflexivity | ].
  apply lifted_refines_one; try assumption.
  - unfold is_allocated. simpl. rewrite Pos.eqb_refl. reflexivity.
  - apply
      (related_closure _ _ _ _ _ _ unnamed_function_slot
         [(unnamed_function_slot, code)] vs vs);
      [ reflexivity | simpl; rewrite Pos.eqb_refl; reflexivity | reflexivity
      | | left; split;
          [ exact I | unfold set_bound; simpl; rewrite Pos.eqb_refl; exact I ]
      ].
    unfold envs_free_of in Henvs.
    apply andb_true_iff in Henvs as [Henvs _].
    apply andb_true_iff in Henvs as [Hvars_free _].
    apply Forall2_same. intros w Hw. apply closed_value_related_except.
    + exact
        (eval_value_slots_closed h vars value_slots vs (proj1 (proj2 Hclosed))
           Hvs w Hw).
    + exact
        (eval_value_slots_avoid _ vars value_slots vs Hvars_free Hslots_avoid
           Hvs w Hw).
Qed.

(* The bindings of a static set, or of the rest of one, bind each symbol
   to the closure of the set at its slot, and nothing else. *)
Lemma eval_bind_closures (vars : env)
    (all_symbols : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple))
    (vs : list (value_slot * value)) :
  eval_value_slots vars value_slots = Some vs ->
  ∀ (syms : list (function_slot * symbol)) (st : static_env),
    (∀ (slot : function_slot) (s : symbol),
      (slot, s) ∈ syms -> lookup slot decls <> None) ->
    distinct (map snd syms) = true ->
    (∀ t : symbol, t ∈ map snd syms -> lookup t (static_values st) = None) ->
    ∃ st1 : static_env,
      eval_static_bindings vars
        (map
           (fun '(slot, s) =>
              Bind_symbol s
                (Static_const_closure_of_set slot all_symbols decls
                   value_slots))
           syms)
        st
      = Some st1
      ∧ static_code st = static_code st1
      ∧ (∀ (slot : function_slot) (s : symbol),
          (slot, s) ∈ syms ->
          lookup s (static_values st1)
          = Some (Static_closure_of_set slot all_symbols decls vs))
      ∧ (∀ t : symbol,
          t ∉ map snd syms ->
          lookup t (static_values st1) = lookup t (static_values st)).
Proof.
  intros Hvs.
  induction syms as [ | [slot s] syms IH];
    intros st Hslots Hdistinct Hunbound.
  - exists st. split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ intros slot s [ ] | reflexivity ].
  - simpl in Hdistinct. apply andb_true_iff in Hdistinct as [Hfresh Hdistinct].
    apply negb_true_iff in Hfresh.
    assert (Hs_not : s ∉ map snd syms).
    { intros Hin. apply among_In in Hin. rewrite Hin in Hfresh.
      discriminate. }
    pose
      (st2 :=
         bind_symbol st s (Static_closure_of_set slot all_symbols decls vs)).
    destruct (IH st2) as (st1 & Heval & Hcode & Hbound & Hkept).
    + intros slot0 s0 Hin. apply (Hslots slot0 s0). right. exact Hin.
    + exact Hdistinct.
    + intros t Ht. simpl.
      destruct (Pos.eqb_spec t s) as [-> | _]; [ contradiction | ].
      apply Hunbound. right. exact Ht.
    + exists st1. split; [ | split; [ exact Hcode | split ] ].
      * simpl. rewrite (Hunbound s (or_introl eq_refl)).
        destruct (lookup slot decls) as [code | ] eqn:Hslot;
          [ | exfalso; exact (Hslots slot s (or_introl eq_refl) Hslot) ].
        simpl. rewrite Hvs. simpl. exact Heval.
      * intros slot0 s0 [[= <- <-] | Hin]; [ | exact (Hbound slot0 s0 Hin) ].
        rewrite (Hkept s Hs_not). simpl. rewrite Pos.eqb_refl. reflexivity.
      * intros t Ht. rewrite (Hkept t (fun Hin => Ht (or_intror Hin))).
        simpl. destruct (Pos.eqb_spec t s) as [-> | _]; [ | reflexivity ].
        exfalso. apply Ht. left. reflexivity.
Qed.

(* Lifting a closure of a set of several functions. *)
Theorem lift_closure_of_set : lift_closure_of_set_statement.
Proof.
  intros x x_kind slot s syms decls value_slots body vars conts statics h D
    Hslot
    Hkeys Hdistinct Hunbound Hslots_avoid Hbody_avoids Hbody_binds Henvs
    Hheap_free Hclosed.
  assert (Hdecls :
            ∀ (slot0 : function_slot) (s0 : symbol),
              (slot0, s0) ∈ syms -> lookup slot0 decls <> None).
  { intros slot0 s0 Hin Hnone. apply lookup_none_not_in in Hnone.
    apply Hnone. pose proof (in_map fst _ _ Hin) as Hin_keys.
    rewrite Hkeys in Hin_keys. exact Hin_keys. }
  destruct (lookup slot decls) as [code | ] eqn:Hcode;
    [ | exfalso; exact (Hdecls slot s (lookup_in _ _ _ Hslot) Hcode) ].
  destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs.
  2: { apply heap_refines_undefined. rewrite heap_step_let.
       cbn [heap_exec_named eval_named]. rewrite Hcode. simpl. rewrite Hvs.
       reflexivity. }
  destruct (kind_eqb Value x_kind) eqn:Hkind.
  2: { apply heap_refines_undefined. rewrite heap_step_let.
       cbn [heap_exec_named eval_named]. rewrite Hcode. cbn [is_some].
       rewrite Hvs. cbn [kind_of_value]. rewrite Hkind. reflexivity. }
  unfold kind_eqb in Hkind.
  destruct (kind_eq_dec Value x_kind) as [<- | _];
    [ clear Hkind | discriminate ].
  pose (x_kind := Value).
  destruct
    (eval_bind_closures vars syms decls value_slots vs Hvs syms statics
       Hdecls Hdistinct Hunbound)
    as (statics1 & Heval & Hcode1 & Hbound & Hkept).
  apply
    (heap_refines_after_source_step _
       (Heap_state body ((x, Value_closure_of_set slot decls vs) :: vars)
          conts statics h)).
  { rewrite heap_step_let. cbn [heap_exec_named eval_named]. rewrite Hcode.
    simpl. rewrite Hvs. reflexivity. }
  apply
    (heap_refines_after_target_step _ _
       (Heap_state (Let x x_kind (Simple (Symbol s)) body) vars conts statics1
          h)).
  { simpl. unfold bind_closures. rewrite Heval. reflexivity. }
  apply
    (heap_refines_after_target_step _ _
       (Heap_state body ((x, Value_symbol s) :: vars) conts statics1 h));
    [ reflexivity | ].
  pose proof (Hbound slot s (lookup_in _ _ _ Hslot)) as Hs.
  apply (lifted_refines D); try assumption.
  - intros t Ht. apply Hkept. intros Hin. unfold symbol_kept in Ht.
    simpl in Ht. apply among_In in Hin. rewrite Hin in Ht. discriminate.
  - intros t Ht. unfold symbol_kept in Ht. simpl in Ht.
    apply negb_false_iff in Ht. apply among_In in Ht.
    split; [ exact (Hunbound t Ht) | ].
    apply in_map_iff in Ht as ([slot0 t0] & <- & Hin). simpl.
    unfold is_allocated. simpl. rewrite (Hbound slot0 t0 Hin).
    destruct (lookup slot0 decls) eqn:Hslot0; [ reflexivity | ].
    exfalso. exact (Hdecls slot0 t0 Hin Hslot0).
  - apply (related_closure _ _ _ _ _ _ slot decls vs vs);
      [ simpl; rewrite Hcode; reflexivity
      | simpl; rewrite Hs; simpl; rewrite Hcode; reflexivity | reflexivity
      | | left; split; [ exact I | ] ].
    + unfold envs_free_of in Henvs.
      apply andb_true_iff in Henvs as [Henvs _].
      apply andb_true_iff in Henvs as [Hvars_free _].
      apply Forall2_same. intros w Hw. apply closed_value_related_except.
      * exact
          (eval_value_slots_closed h vars value_slots vs
             (proj1 (proj2 Hclosed)) Hvs w Hw).
      * exact
          (eval_value_slots_avoid _ vars value_slots vs Hvars_free
             Hslots_avoid Hvs w Hw).
    + unfold set_bound. rewrite Hs. split; [ exact Hkeys | exact Hbound ].
Qed.
