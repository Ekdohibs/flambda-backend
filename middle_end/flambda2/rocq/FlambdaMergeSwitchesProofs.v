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

(* Proofs of the statements of FlambdaMergeSwitchesSpec.v. Nothing here
   needs review.

   The refinement is by the steps of the machine: where the [Switch]
   has a step it goes to an arm, which the test has found to be the
   merged application with a constant in place of the variable; the
   merged term binds the variable to that constant in the steps of
   its [Let]s, and then the two apply the same continuation to the
   same values. With a way for each argument, the [Let]s bind each
   parameter to the constant that the arm has there, the first
   argument first, and leave an environment in which the merged
   arguments have the values of the arguments of the arm
   ([bind_ways_equivalent]). The rest is the recursion of
   FlambdaNormaliseProofs.collapse_switches_refines, with the
   continuations that are known as a parameter. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec
  FlambdaContextProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaMergeSwitchesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The values of the merged arguments                                       *)
(* ======================================================================== *)

(* Under a binding of the variable to a constant an argument has the
   value that it has with the constant in place of the variable, in any
   environment that binds the other variables alike. *)
Lemma eval_simple_constant_for (v : variable) (c : const) (vars vars' : env)
    (s : simple) :
  (∀ x : variable, x <> v -> lookup x vars' = lookup x vars) ->
  eval_simple ((v, Value_const c) :: vars') s
  = eval_simple vars (constant_for v c s).
Proof.
  intros Hagree. destruct s as [x | sym | c']; simpl; try reflexivity.
  destruct (Pos.eqb_spec x v) as [-> | Hne]; [ reflexivity | ].
  simpl. apply Hagree, Hne.
Qed.

Lemma eval_simples_constant_for (v : variable) (c : const) (vars vars' : env)
    (args : list simple) :
  (∀ x : variable, x <> v -> lookup x vars' = lookup x vars) ->
  eval_simples ((v, Value_const c) :: vars') args
  = eval_simples vars (map (constant_for v c) args).
Proof.
  intros Hagree. induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite (eval_simple_constant_for v c vars vars' s Hagree), IH.
  reflexivity.
Qed.

(* So the merged application steps as the arm does. *)
Lemma merged_steps_as_arm (v : variable) (c : const) (merged : apply_cont)
    (vars vars' : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  (∀ x : variable, x <> v -> lookup x vars' = lookup x vars) ->
  step
    (State (Apply_cont merged) ((v, Value_const c) :: vars') conts statics hp
       rd)
  = step
      (State
         (Apply_cont
            (Apply_cont_expr (ac_continuation merged)
               (map (constant_for v c) (ac_args merged))
               (ac_trap_action merged)))
         vars conts statics hp rd).
Proof.
  intros Hagree. rewrite !step_apply_cont. simpl.
  rewrite (eval_simples_constant_for v c vars vars' _ Hagree). reflexivity.
Qed.

(* The arm that a [Switch] goes to is one that the test has looked at. *)
Lemma arm_is_merged_found (how : merge_by) (v : variable)
    (merged : apply_cont) (arms : list (Z * apply_cont)) (i : Z)
    (ac : apply_cont) :
  forallb (arm_is_merged how v merged) arms = true ->
  find_arm i arms = Some ac ->
  ∃ c : const,
    constant_of_arm how i = Some c
    ∧ ac
      = Apply_cont_expr (ac_continuation merged)
          (map (constant_for v c) (ac_args merged))
          (ac_trap_action merged).
Proof.
  intros Hall Hfind. apply find_arm_in in Hfind.
  rewrite forallb_forall in Hall. specialize (Hall (i, ac) Hfind).
  unfold arm_is_merged in Hall. simpl in Hall.
  destruct (constant_of_arm how i) as [c | ]; [ | discriminate ].
  exists c. split; [ reflexivity | ].
  destruct (apply_cont_eq_dec ac _) as [Heq | ]; [ exact Heq | discriminate ].
Qed.

Lemma naked_immediate_of_value_some (w : value) (i : Z) :
  naked_immediate_of_value w = Some i -> w = Value_const (Naked_immediate i).
Proof.
  destruct w; simpl; try discriminate.
  match goal with c : const |- _ => destruct c; simpl; try discriminate end.
  intros [= ->]. reflexivity.
Qed.

(* ======================================================================== *)
(* The steps of the new [Let]s                                              *)
(* ======================================================================== *)

Lemma step_bind_tagged (v : variable) (i : Z) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  lookup v vars = Some (Value_const (Naked_immediate i)) ->
  in_range (int_width Standard_naked_immediate) i = true ->
  step (State (bind_tagged v body) vars conts statics hp rd)
  = Step
      (State body ((v, Value_const (Tagged_immediate i)) :: vars) conts
         statics hp rd).
Proof.
  intros Hv Hrange. unfold bind_tagged.
  rewrite
    (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
       [Value_const (Naked_immediate i)]);
    [ | intros q; discriminate
      | cbn [eval_simples eval_simple]; rewrite Hv; reflexivity
      | intros l [Hl | [ ]]; discriminate Hl ].
  cbn [eval_prim eval_unary_primitive int_of_value].
  rewrite Hrange. reflexivity.
Qed.

Lemma step_bind_negated (v : variable) (i : Z) (c : const) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  lookup v vars = Some (Value_const (Tagged_immediate i)) ->
  constant_of_arm By_tagging_and_negating i = Some c ->
  step (State (bind_negated v body) vars conts statics hp rd)
  = Step (State body ((v, Value_const c) :: vars) conts statics hp rd).
Proof.
  intros Hv Hc. unfold bind_negated.
  rewrite
    (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
       [Value_const (Tagged_immediate i)]);
    [ | intros q; discriminate
      | cbn [eval_simples eval_simple]; rewrite Hv; reflexivity
      | intros l [Hl | [ ]]; discriminate Hl ].
  cbn [eval_prim eval_unary_primitive int_of_value].
  unfold constant_of_arm in Hc.
  destruct (Z.eqb_spec i 0) as [-> | _];
    [ injection Hc as <-; reflexivity | ].
  destruct (Z.eqb_spec i 1) as [-> | _];
    [ injection Hc as <-; reflexivity | discriminate ].
Qed.

(* ======================================================================== *)
(* The refinement                                                           *)
(* ======================================================================== *)

Lemma lookup_other_variable (x v : variable) (w : value) (vars : env) :
  x <> v -> lookup x ((v, w) :: vars) = lookup x vars.
Proof.
  intros Hne. simpl. destruct (Pos.eqb_spec x v); [ contradiction | ].
  reflexivity.
Qed.

Lemma merge_by_tagging_refines (v : variable) (merged : apply_cont)
    (arms : list (Z * apply_cont)) :
  forallb (arm_is_merged By_tagging v merged) arms = true ->
  expr_refines (Switch (Var v) arms) (bind_tagged v (Apply_cont merged)).
Proof.
  intros Hall vars conts statics hp rd.
  destruct (step_switch_cases (Var v) arms vars conts statics hp rd)
    as [Hwrong | (w & i & ac & Hv & Hi & Hfind & Hstep)];
    [ apply step_undefined_refines, Hwrong | ].
  apply naked_immediate_of_value_some in Hi. subst w.
  destruct (arm_is_merged_found _ _ _ _ _ _ Hall Hfind) as (c & Hc & ->).
  unfold constant_of_arm in Hc.
  destruct (in_range (int_width Standard_naked_immediate) i) eqn:Hrange;
    [ | discriminate ].
  injection Hc as <-.
  apply equivalent_refines.
  eapply equivalent_trans; [ apply step_equivalent, Hstep | ].
  apply equivalent_sym.
  eapply equivalent_trans;
    [ apply step_equivalent, (step_bind_tagged v i); [ exact Hv | exact Hrange ]
    | ].
  apply step_eq_equivalent, merged_steps_as_arm. reflexivity.
Qed.

Lemma merge_by_negating_refines (v : variable) (merged : apply_cont)
    (arms : list (Z * apply_cont)) :
  forallb (arm_is_merged By_tagging_and_negating v merged) arms = true ->
  expr_refines (Switch (Var v) arms)
    (bind_tagged v (bind_negated v (Apply_cont merged))).
Proof.
  intros Hall vars conts statics hp rd.
  destruct (step_switch_cases (Var v) arms vars conts statics hp rd)
    as [Hwrong | (w & i & ac & Hv & Hi & Hfind & Hstep)];
    [ apply step_undefined_refines, Hwrong | ].
  apply naked_immediate_of_value_some in Hi. subst w.
  destruct (arm_is_merged_found _ _ _ _ _ _ Hall Hfind) as (c & Hc & ->).
  assert (Hrange : in_range (int_width Standard_naked_immediate) i = true).
  { unfold constant_of_arm in Hc.
    destruct (Z.eqb_spec i 0) as [-> | _]; [ reflexivity | ].
    destruct (Z.eqb_spec i 1) as [-> | _]; [ reflexivity | discriminate ]. }
  apply equivalent_refines.
  eapply equivalent_trans; [ apply step_equivalent, Hstep | ].
  apply equivalent_sym.
  eapply equivalent_trans;
    [ apply step_equivalent, (step_bind_tagged v i); [ exact Hv | exact Hrange ]
    | ].
  eapply equivalent_trans;
    [ apply step_equivalent, (step_bind_negated v i c);
      [ simpl; rewrite Pos.eqb_refl; reflexivity | exact Hc ]
    | ].
  apply step_eq_equivalent, merged_steps_as_arm.
  intros x Hne. apply lookup_other_variable, Hne.
Qed.

(* The variable passed as it is: the value that it has is the constant
   of the arm. *)
Lemma eval_simples_constant_held (v : variable) (c : const) (vars : env)
    (args : list simple) :
  lookup v vars = Some (Value_const c) ->
  eval_simples vars (map (constant_for v c) args) = eval_simples vars args.
Proof.
  intros Hv. induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite IH. destruct s as [x | sym | c']; simpl; try reflexivity.
  destruct (Pos.eqb_spec x v) as [-> | Hne]; [ | reflexivity ].
  simpl. rewrite Hv. reflexivity.
Qed.

Lemma merge_as_it_is_refines (v : variable) (merged : apply_cont)
    (arms : list (Z * apply_cont)) :
  forallb (arm_is_merged As_it_is v merged) arms = true ->
  expr_refines (Switch (Var v) arms) (Apply_cont merged).
Proof.
  intros Hall vars conts statics hp rd.
  destruct (step_switch_cases (Var v) arms vars conts statics hp rd)
    as [Hwrong | (w & i & ac & Hv & Hi & Hfind & Hstep)];
    [ apply step_undefined_refines, Hwrong | ].
  apply naked_immediate_of_value_some in Hi. subst w.
  destruct (arm_is_merged_found _ _ _ _ _ _ Hall Hfind) as (c & Hc & ->).
  injection Hc as <-.
  apply equivalent_refines.
  eapply equivalent_trans; [ apply step_equivalent, Hstep | ].
  apply step_eq_equivalent. rewrite !step_apply_cont. simpl.
  rewrite (eval_simples_constant_held v (Naked_immediate i) vars _ Hv).
  destruct merged. reflexivity.
Qed.

(* The arithmetic. An argument has the value that it has with the
   constant in place of the variable, in any environment in which the
   variable holds the constant and the other variables are bound
   alike. *)
Lemma eval_simples_constant_bound (v : variable) (c : const)
    (vars vars' : env) (args : list simple) :
  lookup v vars' = Some (Value_const c) ->
  (∀ x : variable, x <> v -> lookup x vars' = lookup x vars) ->
  eval_simples vars' args = eval_simples vars (map (constant_for v c) args).
Proof.
  intros Hv Hagree. induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite IH. destruct s as [x | sym | c']; simpl; try reflexivity.
  destruct (Pos.eqb_spec x v) as [-> | Hne]; simpl;
    [ rewrite Hv | rewrite (Hagree x Hne) ]; reflexivity.
Qed.

Lemma merged_steps_as_arm_bound (v : variable) (c : const)
    (merged : apply_cont) (vars vars' : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  lookup v vars' = Some (Value_const c) ->
  (∀ x : variable, x <> v -> lookup x vars' = lookup x vars) ->
  step (State (Apply_cont merged) vars' conts statics hp rd)
  = step
      (State
         (Apply_cont
            (Apply_cont_expr (ac_continuation merged)
               (map (constant_for v c) (ac_args merged))
               (ac_trap_action merged)))
         vars conts statics hp rd).
Proof.
  intros Hv Hagree. rewrite !step_apply_cont. simpl.
  rewrite (eval_simples_constant_bound v c vars vars' _ Hv Hagree).
  reflexivity.
Qed.

(* What a step of the arithmetic makes of an integer of the width is
   one. *)
Lemma arithmetic_step_value_in_range (standard : standard_int)
    (arith : arithmetic_step) (a a' : Z) :
  arithmetic_step_value (int_width standard) arith a = Some a' ->
  in_range (int_width standard) a' = true.
Proof.
  destruct arith as [c | c | c]; cbn [arithmetic_step_value];
    destruct (in_range (int_width standard) c); try discriminate;
    intros [= <-]; apply wrap_in_range, int_width_positive.
Qed.

(* The [Let] of a step of the arithmetic binds the variable to what
   the step makes of the integer that it holds. *)
Lemma step_bind_arithmetic_step (standard : standard_int) (v : variable)
    (arith : arithmetic_step) (a a' : Z) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  lookup v vars = Some (Value_const (const_of_int standard a)) ->
  in_range (int_width standard) a = true ->
  arithmetic_step_value (int_width standard) arith a = Some a' ->
  step
    (State (bind_arithmetic_step standard v arith body) vars conts statics hp
       rd)
  = Step
      (State body ((v, Value_const (const_of_int standard a')) :: vars) conts
         statics hp rd).
Proof.
  intros Hv Hrange Hvalue.
  assert
    (Hint :
      ∀ c : Z,
        in_range (int_width standard) c = true ->
        int_of_value standard (Value_const (const_of_int standard c))
        = Some c).
  { intros c Hc. destruct standard; simpl in *; rewrite Hc; reflexivity. }
  assert
    (Hkind :
      ∀ c : Z,
        kind_eqb (kind_of_value (Value_const (const_of_int standard c)))
          (kind_of_standard_int standard)
        = true).
  { intros c. destruct standard; reflexivity. }
  assert
    (Hnone :
      ∀ (c1 c2 : Z) (l : location),
        Value_location l
        ∉ [Value_const (const_of_int standard c1);
           Value_const (const_of_int standard c2)]).
  { intros c1 c2 l [Hl | [Hl | [ ]]]; destruct standard; discriminate Hl. }
  unfold bind_arithmetic_step.
  destruct arith as [c | c | c]; cbn [arithmetic_step_value] in Hvalue;
    destruct (in_range (int_width standard) c) eqn:Hc; try discriminate;
    injection Hvalue as <-; cbn beta iota zeta;
    [ rewrite
        (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
           [Value_const (const_of_int standard a);
            Value_const (const_of_int standard c)])
    | rewrite
        (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
           [Value_const (const_of_int standard c);
            Value_const (const_of_int standard a)])
    | rewrite
        (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
           [Value_const (const_of_int standard c);
            Value_const (const_of_int standard a)]) ];
    try (intros q; discriminate);
    try (cbn [eval_simples eval_simple]; rewrite Hv; reflexivity);
    try apply Hnone;
    cbn [eval_prim eval_binary_primitive];
    rewrite (Hint a Hrange), (Hint c Hc); cbn [eval_int_arith];
    rewrite Hkind; reflexivity.
Qed.

(* So do the [Let]s of the steps, one after the other. The
   environment that they leave binds the other variables as before. *)
Lemma bind_arithmetic_steps_equivalent (standard : standard_int)
    (v : variable) (body : expr) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) (steps : list arithmetic_step) :
  ∀ (a a' : Z) (vars : env),
    lookup v vars = Some (Value_const (const_of_int standard a)) ->
    in_range (int_width standard) a = true ->
    arithmetic_value (int_width standard) steps a = Some a' ->
    ∃ vars' : env,
      lookup v vars' = Some (Value_const (const_of_int standard a'))
      ∧ (∀ x : variable, x <> v -> lookup x vars' = lookup x vars)
      ∧ equivalent
          (State (bind_arithmetic_steps standard v steps body) vars conts
             statics hp rd)
          (State body vars' conts statics hp rd).
Proof.
  induction steps as [ | arith steps IH]; intros a a' vars Hv Hrange Hvalue.
  - injection Hvalue as <-. exists vars.
    split; [ exact Hv | split; [ reflexivity | apply equivalent_refl ] ].
  - cbn [arithmetic_value] in Hvalue.
    destruct (arithmetic_step_value (int_width standard) arith a)
      as [a1 | ] eqn:Hstep; [ | discriminate ].
    destruct
      (IH a1 a' ((v, Value_const (const_of_int standard a1)) :: vars))
      as (vars' & Hv' & Hagree & Hequiv).
    + simpl. rewrite Pos.eqb_refl. reflexivity.
    + exact (arithmetic_step_value_in_range standard arith a a1 Hstep).
    + exact Hvalue.
    + exists vars'. split; [ exact Hv' | split ].
      * intros x Hne. rewrite (Hagree x Hne).
        apply lookup_other_variable, Hne.
      * eapply equivalent_trans; [ | exact Hequiv ].
        apply step_equivalent.
        exact
          (step_bind_arithmetic_step standard v arith a a1 _ vars conts
             statics hp rd Hv Hrange Hstep).
Qed.

Lemma merge_by_arithmetic_refines (kind : arithmetic_kind) (offset slope : Z)
    (v : variable) (merged : apply_cont) (arms : list (Z * apply_cont)) :
  forallb (arm_is_merged (By_arithmetic kind offset slope) v merged) arms
  = true ->
  expr_refines (Switch (Var v) arms)
    (bind_arithmetic kind offset slope v (Apply_cont merged)).
Proof.
  intros Hall vars conts statics hp rd.
  destruct (step_switch_cases (Var v) arms vars conts statics hp rd)
    as [Hwrong | (w & i & ac & Hv & Hi & Hfind & Hstep)];
    [ apply step_undefined_refines, Hwrong | ].
  apply naked_immediate_of_value_some in Hi. subst w.
  destruct (arm_is_merged_found _ _ _ _ _ _ Hall Hfind) as (c & Hc & ->).
  cbn [constant_of_arm] in Hc. cbv zeta in Hc.
  destruct (in_range (int_width (standard_int_of_arithmetic_kind kind)) i)
    eqn:Hrange; [ | discriminate ].
  destruct (arithmetic_value _ _ i) as [a | ] eqn:Hvalue; [ | discriminate ].
  injection Hc as <-.
  apply equivalent_refines.
  eapply equivalent_trans; [ apply step_equivalent, Hstep | ].
  apply equivalent_sym.
  unfold bind_arithmetic. cbv zeta.
  destruct kind; cbn [standard_int_of_arithmetic_kind] in *.
  - eapply equivalent_trans;
      [ apply step_equivalent, (step_bind_tagged v i);
        [ exact Hv | exact Hrange ]
      | ].
    destruct
      (bind_arithmetic_steps_equivalent Standard_tagged_immediate v
         (Apply_cont merged) conts statics hp rd
         (arithmetic_steps (int_width Standard_tagged_immediate) offset slope)
         i a ((v, Value_const (Tagged_immediate i)) :: vars))
      as (vars' & Hv' & Hagree & Hequiv);
      [ simpl; rewrite Pos.eqb_refl; reflexivity | exact Hrange
      | exact Hvalue | ].
    eapply equivalent_trans; [ exact Hequiv | ].
    apply step_eq_equivalent, merged_steps_as_arm_bound; [ exact Hv' | ].
    intros x Hne. rewrite (Hagree x Hne). apply lookup_other_variable, Hne.
  - destruct
      (bind_arithmetic_steps_equivalent Standard_naked_immediate v
         (Apply_cont merged) conts statics hp rd
         (arithmetic_steps (int_width Standard_naked_immediate) offset slope)
         i a vars)
      as (vars' & Hv' & Hagree & Hequiv);
      [ exact Hv | exact Hrange | exact Hvalue | ].
    eapply equivalent_trans; [ exact Hequiv | ].
    apply step_eq_equivalent, merged_steps_as_arm_bound;
      [ exact Hv' | exact Hagree ].
Qed.

Lemma merge_switch_refines (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  expr_refines (Switch scrutinee arms) (merge_switch scrutinee arms).
Proof.
  unfold merge_switch.
  destruct scrutinee as [v | sym | c]; try apply expr_refines_refl.
  destruct arms as [ | [i ac] [ | [i' ac'] others]];
    try apply expr_refines_refl.
  cbv zeta.
  destruct (var_in_simples v _); [ | apply expr_refines_refl ].
  destruct (forallb (arm_is_merged By_tagging v _) _) eqn:Htagging;
    [ apply merge_by_tagging_refines, Htagging | ].
  destruct (forallb (arm_is_merged By_tagging_and_negating v _) _)
    eqn:Hnegating; [ apply merge_by_negating_refines, Hnegating | ].
  destruct (forallb (arm_is_merged As_it_is v _) _) eqn:Has_it_is;
    [ apply merge_as_it_is_refines, Has_it_is | ].
  destruct (arithmetic_of_arms _ _ _ _) as [[[kind offset] slope] | ];
    [ | apply expr_refines_refl ].
  destruct (_ && _) eqn:Harithmetic; [ | apply expr_refines_refl ].
  apply andb_prop in Harithmetic as [_ Harithmetic].
  apply merge_by_arithmetic_refines, Harithmetic.
Qed.

(* ======================================================================== *)
(* The ways chosen for each argument                                        *)
(* ======================================================================== *)

Lemma step_bind_tagged_to (v x : variable) (i : Z) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  lookup v vars = Some (Value_const (Naked_immediate i)) ->
  in_range (int_width Standard_naked_immediate) i = true ->
  step (State (bind_tagged_to v x body) vars conts statics hp rd)
  = Step
      (State body ((x, Value_const (Tagged_immediate i)) :: vars) conts
         statics hp rd).
Proof.
  intros Hv Hrange. unfold bind_tagged_to.
  rewrite
    (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
       [Value_const (Naked_immediate i)]);
    [ | intros q; discriminate
      | cbn [eval_simples eval_simple]; rewrite Hv; reflexivity
      | intros l [Hl | [ ]]; discriminate Hl ].
  cbn [eval_prim eval_unary_primitive int_of_value].
  rewrite Hrange. reflexivity.
Qed.

Lemma step_bind_same_to (v x : variable) (i : Z) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  lookup v vars = Some (Value_const (Naked_immediate i)) ->
  step (State (bind_same_to v x body) vars conts statics hp rd)
  = Step
      (State body ((x, Value_const (Naked_immediate i)) :: vars) conts
         statics hp rd).
Proof.
  intros Hv. unfold bind_same_to. rewrite step_let_simple.
  cbn [eval_named eval_simple]. rewrite Hv. reflexivity.
Qed.

(* A way that binds the variable that it is given. *)
Definition way_binds (how : merge_by) : bool :=
  match how with
  | As_it_is => false
  | By_tagging | By_tagging_and_negating | By_arithmetic _ _ _ => true
  end.

(* The [Let]s of a way bind the variable to the constant of the arm,
   and leave the other variables as they are. *)
Lemma bind_way_equivalent (how : merge_by) (v x : variable) (i : Z)
    (c : const) (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  lookup v vars = Some (Value_const (Naked_immediate i)) ->
  constant_of_arm how i = Some c ->
  ∃ vars' : env,
    (∀ y : variable, y <> x -> lookup y vars' = lookup y vars)
    ∧ (way_binds how = true -> lookup x vars' = Some (Value_const c))
    ∧ equivalent (State (bind_way how v x body) vars conts statics hp rd)
        (State body vars' conts statics hp rd).
Proof.
  intros Hv Hc. destruct how as [ | | | kind offset slope].
  - cbn [constant_of_arm] in Hc.
    destruct (in_range (int_width Standard_naked_immediate) i) eqn:Hrange;
      [ | discriminate ].
    injection Hc as <-.
    exists ((x, Value_const (Tagged_immediate i)) :: vars).
    split; [ intros y Hne; apply lookup_other_variable, Hne | ].
    split; [ intros _; simpl; rewrite Pos.eqb_refl; reflexivity | ].
    apply step_equivalent, step_bind_tagged_to; assumption.
  - assert (Hrange : in_range (int_width Standard_naked_immediate) i = true).
    { cbn [constant_of_arm] in Hc.
      destruct (Z.eqb_spec i 0) as [-> | _]; [ reflexivity | ].
      destruct (Z.eqb_spec i 1) as [-> | _]; [ reflexivity | discriminate ]. }
    exists
      ((x, Value_const c) :: (x, Value_const (Tagged_immediate i)) :: vars).
    split.
    { intros y Hne.
      rewrite (lookup_other_variable y x _ _ Hne).
      apply lookup_other_variable, Hne. }
    split; [ intros _; simpl; rewrite Pos.eqb_refl; reflexivity | ].
    cbn [bind_way].
    eapply equivalent_trans;
      [ apply step_equivalent, (step_bind_tagged_to v x i); assumption | ].
    apply step_equivalent, (step_bind_negated x i c);
      [ simpl; rewrite Pos.eqb_refl; reflexivity | exact Hc ].
  - exists vars.
    split; [ reflexivity | split; [ discriminate | apply equivalent_refl ] ].
  - cbn [constant_of_arm] in Hc. cbv zeta in Hc.
    destruct (in_range (int_width (standard_int_of_arithmetic_kind kind)) i)
      eqn:Hrange; [ | discriminate ].
    destruct (arithmetic_value _ _ i) as [a | ] eqn:Hvalue; [ | discriminate ].
    injection Hc as <-.
    cbn [bind_way]. cbv zeta.
    destruct kind; cbn [standard_int_of_arithmetic_kind] in *.
    + destruct
        (bind_arithmetic_steps_equivalent Standard_tagged_immediate x body
           conts statics hp rd
           (arithmetic_steps (int_width Standard_tagged_immediate) offset
              slope)
           i a ((x, Value_const (Tagged_immediate i)) :: vars))
        as (vars' & Hx' & Hagree & Hequiv);
        [ simpl; rewrite Pos.eqb_refl; reflexivity | exact Hrange
        | exact Hvalue | ].
      exists vars'. split.
      { intros y Hne. rewrite (Hagree y Hne).
        apply lookup_other_variable, Hne. }
      split; [ intros _; exact Hx' | ].
      eapply equivalent_trans; [ | exact Hequiv ].
      apply step_equivalent, (step_bind_tagged_to v x i); assumption.
    + destruct
        (bind_arithmetic_steps_equivalent Standard_naked_immediate x body
           conts statics hp rd
           (arithmetic_steps (int_width Standard_naked_immediate) offset
              slope)
           i a ((x, Value_const (Naked_immediate i)) :: vars))
        as (vars' & Hx' & Hagree & Hequiv);
        [ simpl; rewrite Pos.eqb_refl; reflexivity | exact Hrange
        | exact Hvalue | ].
      exists vars'. split.
      { intros y Hne. rewrite (Hagree y Hne).
        apply lookup_other_variable, Hne. }
      split; [ intros _; exact Hx' | ].
      eapply equivalent_trans; [ | exact Hequiv ].
      apply step_equivalent, (step_bind_same_to v x i), Hv.
Qed.

(* The arguments that are the same in every arm mention none of the
   variables of a list: the test of [names_usable]. *)
Definition same_avoid (avoid : list variable) (ways : list argument_way)
    : bool :=
  forallb
    (fun way =>
       match way with
       | Same_in_every_arm s =>
         negb (existsb (fun x => var_in_simple x s) avoid)
       | Computed _ => true
       end)
    ways.

Lemma eval_simple_avoiding (avoid : list variable) (s : simple)
    (vars vars' : env) :
  existsb (fun x => var_in_simple x s) avoid = false ->
  (∀ y : variable, among y avoid = false -> lookup y vars' = lookup y vars) ->
  eval_simple vars' s = eval_simple vars s.
Proof.
  intros Hs Hagree. destruct s as [y | sym | c]; try reflexivity.
  simpl. apply Hagree. unfold among. rewrite <- Hs. clear Hs Hagree.
  induction avoid as [ | x avoid IH]; simpl; [ reflexivity | ].
  rewrite IH, (Pos.eqb_sym y x). reflexivity.
Qed.

(* So the arguments of an arm have the same values in two
   environments that differ on those variables only. *)
Lemma arguments_in_arm_eval (i : Z) (avoid : list variable)
    (vars vars' : env) (ways : list argument_way) :
  ∀ args : list simple,
    arguments_in_arm i ways = Some args ->
    same_avoid avoid ways = true ->
    (∀ y : variable, among y avoid = false -> lookup y vars' = lookup y vars) ->
    eval_simples vars' args = eval_simples vars args.
Proof.
  induction ways as [ | way ways IH]; intros args Hargs Hsame Hagree;
    cbn [arguments_in_arm] in Hargs.
  - injection Hargs as <-. reflexivity.
  - destruct (argument_in_arm i way) as [s | ] eqn:Hs; [ | discriminate ].
    destruct (arguments_in_arm i ways) as [args0 | ]; [ | discriminate ].
    injection Hargs as <-.
    unfold same_avoid in Hsame. cbn [forallb] in Hsame.
    apply andb_prop in Hsame as [Hway Hsame].
    cbn [eval_simples]. rewrite (IH args0 eq_refl Hsame Hagree).
    replace (eval_simple vars' s) with (eval_simple vars s); [ reflexivity | ].
    destruct way as [s0 | how]; cbn [argument_in_arm] in Hs.
    + injection Hs as <-. symmetry.
      apply (eval_simple_avoiding avoid); [ | exact Hagree ].
      apply negb_true_iff, Hway.
    + destruct (constant_of_arm how i); [ | discriminate ].
      injection Hs as <-. reflexivity.
Qed.

(* The [Let]s for the arguments leave an environment in which the
   merged arguments have the values of the arguments of the arm. *)
Lemma bind_ways_equivalent (v : variable) (i : Z) (avoid : list variable)
    (body : expr) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record)
    (names : list variable) :
  ∀ (ways : list argument_way) (args : list simple) (vars : env),
    lookup v vars = Some (Value_const (Naked_immediate i)) ->
    length names = length ways ->
    distinct names = true ->
    among v names = false ->
    (∀ x : variable, among x names = true -> among x avoid = true) ->
    same_avoid avoid ways = true ->
    arguments_in_arm i ways = Some args ->
    ∃ vars' : env,
      (∀ y : variable, among y names = false -> lookup y vars' = lookup y vars)
      ∧ equivalent
          (State (bind_ways v names ways body) vars conts statics hp rd)
          (State body vars' conts statics hp rd)
      ∧ eval_simples vars' (merged_by_ways v names ways)
        = eval_simples vars args.
Proof.
  induction names as [ | x names IH];
    intros [ | way ways] args vars Hv Hlength Hdistinct Hvnames Havoid Hsame
      Hargs;
    try discriminate.
  - injection Hargs as <-. exists vars.
    split; [ reflexivity | split; [ apply equivalent_refl | reflexivity ] ].
  - injection Hlength as Hlength.
    cbn [distinct] in Hdistinct.
    apply andb_prop in Hdistinct as [Hxnames Hdistinct].
    apply negb_true_iff in Hxnames.
    unfold among in Hvnames. cbn [existsb] in Hvnames.
    apply orb_false_elim in Hvnames as [Hvx Hvnames].
    apply Pos.eqb_neq in Hvx.
    assert (Hxavoid : among x avoid = true).
    { apply Havoid. unfold among. cbn [existsb]. rewrite Pos.eqb_refl.
      reflexivity. }
    assert
      (Havoid' :
        ∀ y : variable, among y names = true -> among y avoid = true).
    { intros y Hy. apply Havoid. unfold among in *. cbn [existsb].
      rewrite Hy. apply orb_true_r. }
    assert
      (Hoff :
        ∀ (vars1 vars2 : env),
          (∀ y : variable,
             among y names = false -> lookup y vars2 = lookup y vars1) ->
          ∀ y : variable,
            among y avoid = false -> lookup y vars2 = lookup y vars1).
    { intros vars1 vars2 Hagree y Hy. apply Hagree.
      destruct (among y names) eqn:Hin; [ | reflexivity ].
      rewrite (Havoid' y Hin) in Hy. discriminate. }
    pose proof Hsame as Hsame'.
    unfold same_avoid in Hsame'. cbn [forallb] in Hsame'.
    apply andb_prop in Hsame' as [Hway Hrest].
    cbn [arguments_in_arm] in Hargs.
    destruct (argument_in_arm i way) as [s0 | ] eqn:Hs0; [ | discriminate ].
    destruct (arguments_in_arm i ways) as [args0 | ] eqn:Hargs0;
      [ | discriminate ].
    injection Hargs as <-.
    destruct way as [s | how]; cbn [argument_in_arm] in Hs0.
    + injection Hs0 as <-.
      destruct
        (IH ways args0 vars Hv Hlength Hdistinct Hvnames Havoid' Hrest
           Hargs0)
        as (vars' & Hagree & Hequiv & Heval).
      exists vars'. split.
      { intros y Hy. apply Hagree. unfold among in *. cbn [existsb] in Hy.
        apply orb_false_elim in Hy as [_ Hy]. exact Hy. }
      split; [ exact Hequiv | ].
      cbn [merged_by_ways eval_simples]. rewrite Heval.
      rewrite (eval_simple_avoiding avoid s vars vars');
        [ reflexivity | apply negb_true_iff, Hway
        | exact (Hoff vars vars' Hagree) ].
    + destruct (constant_of_arm how i) as [c | ] eqn:Hc; [ | discriminate ].
      injection Hs0 as <-.
      destruct
        (bind_way_equivalent how v x i c (bind_ways v names ways body) vars
           conts statics hp rd Hv Hc)
        as (vars1 & Hagree1 & Hx1 & Hequiv1).
      assert (Hv1 : lookup v vars1 = Some (Value_const (Naked_immediate i))).
      { rewrite (Hagree1 v Hvx). exact Hv. }
      destruct
        (IH ways args0 vars1 Hv1 Hlength Hdistinct Hvnames Havoid' Hrest
           Hargs0)
        as (vars' & Hagree & Hequiv & Heval).
      exists vars'. split.
      { intros y Hy. unfold among in Hy. cbn [existsb] in Hy.
        apply orb_false_elim in Hy as [Hyx Hy].
        rewrite (Hagree y Hy). apply Hagree1, Pos.eqb_neq, Hyx. }
      split; [ exact (equivalent_trans _ _ _ Hequiv1 Hequiv) | ].
      cbn [bind_ways merged_by_ways eval_simples eval_simple].
      rewrite Heval.
      rewrite
        (arguments_in_arm_eval i avoid vars vars1 ways args0 Hargs0 Hrest).
      2: { intros y Hy. apply Hagree1. intros ->.
           rewrite Hxavoid in Hy. discriminate. }
      replace
        (eval_simple vars'
           match how with
           | As_it_is => Var v
           | By_tagging | By_tagging_and_negating | By_arithmetic _ _ _ =>
             Var x
           end)
        with (Some (Value_const c)); [ reflexivity | ].
      symmetry.
      destruct how as [ | | | kind offset slope]; cbn [eval_simple].
      * rewrite (Hagree x Hxnames). exact (Hx1 eq_refl).
      * rewrite (Hagree x Hxnames). exact (Hx1 eq_refl).
      * rewrite (Hagree v Hvnames), Hv1. cbn [constant_of_arm] in Hc.
        injection Hc as <-. reflexivity.
      * rewrite (Hagree x Hxnames). exact (Hx1 eq_refl).
Qed.

(* The arm that a [Switch] goes to is one that the test has looked at. *)
Lemma arm_has_ways_found (k : continuation) (ways : list argument_way)
    (arms : list (Z * apply_cont)) (i : Z) (ac : apply_cont) :
  forallb (arm_has_ways k ways) arms = true ->
  find_arm i arms = Some ac ->
  ∃ args : list simple,
    arguments_in_arm i ways = Some args ∧ ac = Apply_cont_expr k args None.
Proof.
  intros Hall Hfind. apply find_arm_in in Hfind.
  rewrite forallb_forall in Hall. specialize (Hall (i, ac) Hfind).
  unfold arm_has_ways in Hall. simpl in Hall.
  destruct (arguments_in_arm i ways) as [args | ]; [ | discriminate ].
  exists args. split; [ reflexivity | ].
  destruct (apply_cont_eq_dec ac _) as [Heq | ]; [ exact Heq | discriminate ].
Qed.

Lemma merge_switch_by_arguments_refines (names : list variable)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (merged : expr) :
  merge_switch_by_arguments names scrutinee arms = Some merged ->
  expr_refines (Switch scrutinee arms) merged.
Proof.
  unfold merge_switch_by_arguments.
  destruct scrutinee as [v | sym | c]; try discriminate.
  destruct arms as [ | [i0 ac0] [ | arm1 others]]; try discriminate.
  cbv zeta.
  destruct (ways_of_arguments _ _) as [ways | ]; [ | discriminate ].
  destruct (names_usable v names ways && _) eqn:Hok; [ | discriminate ].
  intros [= <-].
  apply andb_prop in Hok as [Husable Hall].
  unfold names_usable in Husable.
  apply andb_prop in Husable as [Husable Hsame].
  apply andb_prop in Husable as [Husable Hvnames].
  apply andb_prop in Husable as [Hlength Hdistinct].
  apply Nat.eqb_eq in Hlength. apply negb_true_iff in Hvnames.
  intros vars conts statics hp rd.
  destruct
    (step_switch_cases (Var v) (((i0, ac0) :: arm1 :: others)) vars conts
       statics hp rd)
    as [Hwrong | (w & i & ac & Hv & Hi & Hfind & Hstep)];
    [ apply step_undefined_refines, Hwrong | ].
  apply naked_immediate_of_value_some in Hi. subst w.
  destruct (arm_has_ways_found _ _ _ _ _ Hall Hfind) as (args & Hargs & ->).
  destruct
    (bind_ways_equivalent v i names
       (Apply_cont
          (Apply_cont_expr (ac_continuation ac0)
             (merged_by_ways v names ways) None))
       conts statics hp rd names ways args vars Hv Hlength Hdistinct Hvnames
       (fun x Hx => Hx) Hsame Hargs)
    as (vars' & _ & Hequiv & Heval).
  apply equivalent_refines.
  eapply equivalent_trans; [ apply step_equivalent, Hstep | ].
  apply equivalent_sym.
  eapply equivalent_trans; [ exact Hequiv | ].
  apply step_eq_equivalent. rewrite !step_apply_cont.
  cbn [ac_args ac_continuation]. rewrite Heval. reflexivity.
Qed.

Lemma merge_switch_knowing_refines (known : known_continuations)
    (scrutinee : simple) (arms : list (Z * apply_cont)) :
  expr_refines (Switch scrutinee arms)
    (merge_switch_knowing known scrutinee arms).
Proof.
  unfold merge_switch_knowing.
  pose proof (merge_switch_refines scrutinee arms) as Hone.
  destruct (merge_switch scrutinee arms); try exact Hone.
  destruct arms as [ | [i ac] others]; [ apply expr_refines_refl | ].
  destruct (lookup (ac_continuation ac) known) as [names | ];
    [ | apply expr_refines_refl ].
  destruct (merge_switch_by_arguments names scrutinee ((i, ac) :: others))
    as [merged | ] eqn:Hmerged;
    [ exact (merge_switch_by_arguments_refines _ _ _ _ Hmerged)
    | apply expr_refines_refl ].
Qed.

Fixpoint merge_switches_knowing_refines (known : known_continuations)
    (e : expr) {struct e} :
  expr_refines e (merge_switches_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl.
  - exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
      (merge_switches_knowing_refines known body)).
  - exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
      (merge_switches_knowing_refines known body)).
  - apply (expr_refines_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (merge_switches_knowing known handler))
         body)).
    + exact (plug_refines
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (merge_switches_knowing_refines known handler)).
    + exact (plug_refines
        (Let_cont_non_recursive_body k
           (Handler params metadata (merge_switches_knowing known handler))
           Hole)
        _ _ (not_many _)
        (merge_switches_knowing_refines ((k, params) :: known) body)).
  - exact (plug_refines (Let_cont_recursive_body handlers Hole) _ _ (not_many _)
      (merge_switches_knowing_refines _ body)).
  - apply merge_switch_knowing_refines.
Qed.

Theorem merge_switches_refines : merge_switches_refines_statement.
Proof. exact (merge_switches_knowing_refines []). Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* A renaming is injective. The lemmas have names of their own, as the
   files that use this one have lemmas that say the same. *)
Lemma renamed_simple_same (r : renaming) (s s' : simple) :
  rename_simple r s = rename_simple r s' -> s = s'.
Proof.
  destruct s, s'; simpl; intros Heq; try discriminate; try exact Heq.
  injection Heq as Heq. apply rename_variable_injective in Heq. congruence.
Qed.

Lemma renamed_simples_same (r : renaming) (args args' : list simple) :
  map (rename_simple r) args = map (rename_simple r) args' -> args = args'.
Proof.
  revert args'.
  induction args as [ | s args IH]; intros [ | s' args']; simpl; intros Heq;
    try discriminate; [ reflexivity | ].
  injection Heq as Hs Hargs.
  apply renamed_simple_same in Hs. apply IH in Hargs. congruence.
Qed.

Lemma renamed_trap_action_same (r : renaming)
    (action action' : option trap_action) :
  option_map (rename_trap_action r) action
  = option_map (rename_trap_action r) action' ->
  action = action'.
Proof.
  destruct action as [[k | k kind] | ], action' as [[k' | k' kind'] | ];
    simpl; intros Heq; try discriminate; try reflexivity;
    injection Heq as Hk; apply rename_continuation_injective in Hk;
    congruence.
Qed.

Lemma renamed_apply_cont_same (r : renaming) (ac ac' : apply_cont) :
  rename_apply_cont r ac = rename_apply_cont r ac' -> ac = ac'.
Proof.
  destruct ac as [k args trap], ac' as [k' args' trap'].
  unfold rename_apply_cont. simpl. intros Heq.
  injection Heq as Hk Hargs Htrap.
  apply rename_continuation_injective in Hk.
  apply renamed_simples_same in Hargs.
  apply renamed_trap_action_same in Htrap. congruence.
Qed.

Lemma forallb_under_map {A B : Type} (f : B -> bool) (g : A -> B)
    (h : A -> bool) (l : list A) :
  (∀ a : A, f (g a) = h a) -> forallb f (map g l) = forallb h l.
Proof.
  intros Hfg. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hfg, IH. reflexivity.
Qed.

(* Each function that the rewrite is made of commutes with a renaming. *)
Lemma constant_for_renamed (r : renaming) (v : variable) (c : const)
    (s : simple) :
  constant_for (rename_variable r v) c (rename_simple r s)
  = rename_simple r (constant_for v c s).
Proof.
  destruct s as [x | sym | c']; simpl; try reflexivity.
  rewrite rename_variable_eqb. destruct (Pos.eqb x v); reflexivity.
Qed.

Lemma constants_for_renamed (r : renaming) (v : variable) (c : const)
    (args : list simple) :
  map (constant_for (rename_variable r v) c) (map (rename_simple r) args)
  = map (rename_simple r) (map (constant_for v c) args).
Proof.
  rewrite !map_map. apply map_ext. intros s. apply constant_for_renamed.
Qed.

Lemma merged_arguments_renamed (r : renaming) (v : variable)
    (args args' : list simple) :
  merged_arguments (rename_variable r v) (map (rename_simple r) args)
    (map (rename_simple r) args')
  = map (rename_simple r) (merged_arguments v args args').
Proof.
  revert args'.
  induction args as [ | s args IH]; intros [ | s' args']; simpl;
    try reflexivity.
  rewrite IH. f_equal.
  destruct (simple_eq_dec s s') as [-> | Hne],
    (simple_eq_dec (rename_simple r _) (rename_simple r _)) as [Heq | Hne'];
    try reflexivity; [ contradiction | ].
  elim Hne. exact (renamed_simple_same r _ _ Heq).
Qed.

Lemma var_in_simples_renamed (r : renaming) (v : variable)
    (args : list simple) :
  var_in_simples (rename_variable r v) (map (rename_simple r) args)
  = var_in_simples v args.
Proof.
  unfold var_in_simples. induction args as [ | s args IH]; simpl;
    [ reflexivity | ].
  rewrite IH. f_equal. destruct s; simpl; try reflexivity.
  apply rename_variable_eqb.
Qed.

Lemma arm_is_merged_renamed (how : merge_by) (r : renaming) (v : variable)
    (merged : apply_cont) (arm : Z * apply_cont) :
  arm_is_merged how (rename_variable r v) (rename_apply_cont r merged)
    (fst arm, rename_apply_cont r (snd arm))
  = arm_is_merged how v merged arm.
Proof.
  unfold arm_is_merged. cbn [fst snd].
  destruct (constant_of_arm how (fst arm)) as [c | ]; [ | reflexivity ].
  set (target :=
    Apply_cont_expr (ac_continuation merged)
      (map (constant_for v c) (ac_args merged)) (ac_trap_action merged)).
  replace
    (Apply_cont_expr (ac_continuation (rename_apply_cont r merged))
       (map (constant_for (rename_variable r v) c)
          (ac_args (rename_apply_cont r merged)))
       (ac_trap_action (rename_apply_cont r merged)))
    with (rename_apply_cont r target).
  - destruct (apply_cont_eq_dec (snd arm) target) as [-> | Hne],
      (apply_cont_eq_dec (rename_apply_cont r _) (rename_apply_cont r _))
        as [Heq | Hne'];
      try reflexivity; [ contradiction | ].
    elim Hne. exact (renamed_apply_cont_same r _ _ Heq).
  - unfold target, rename_apply_cont. cbn [ac_continuation ac_args
      ac_trap_action].
    rewrite constants_for_renamed. reflexivity.
Qed.

Lemma first_difference_renamed (r : renaming) (args args' : list simple) :
  first_difference (map (rename_simple r) args) (map (rename_simple r) args')
  = first_difference args args'.
Proof.
  revert args'.
  induction args as [ | s args IH]; intros [ | s' args']; simpl;
    try reflexivity.
  destruct (simple_eq_dec s s') as [-> | Hne],
    (simple_eq_dec (rename_simple r _) (rename_simple r _)) as [Heq | Hne'].
  - apply IH.
  - contradiction.
  - elim Hne. exact (renamed_simple_same r _ _ Heq).
  - destruct s, s'; reflexivity.
Qed.

Lemma arithmetic_of_arms_renamed (r : renaming) (i i' : Z)
    (args args' : list simple) :
  arithmetic_of_arms i i' (map (rename_simple r) args)
    (map (rename_simple r) args')
  = arithmetic_of_arms i i' args args'.
Proof.
  unfold arithmetic_of_arms. rewrite first_difference_renamed. reflexivity.
Qed.

Lemma bind_arithmetic_steps_renamed (r : renaming) (standard : standard_int)
    (v : variable) (steps : list arithmetic_step) (body : expr) :
  rename_expr r (bind_arithmetic_steps standard v steps body)
  = bind_arithmetic_steps standard (rename_variable r v) steps
      (rename_expr r body).
Proof.
  induction steps as [ | [c | c | c] steps IH]; simpl;
    [ reflexivity | rewrite IH; reflexivity .. ].
Qed.

Lemma bind_arithmetic_renamed (r : renaming) (kind : arithmetic_kind)
    (offset slope : Z) (v : variable) (body : expr) :
  rename_expr r (bind_arithmetic kind offset slope v body)
  = bind_arithmetic kind offset slope (rename_variable r v)
      (rename_expr r body).
Proof.
  unfold bind_arithmetic. cbv zeta.
  destruct kind; [ unfold bind_tagged; simpl | ];
    rewrite bind_arithmetic_steps_renamed; reflexivity.
Qed.

Lemma merge_switch_equivariant (r : renaming) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  merge_switch (rename_simple r scrutinee)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = rename_expr r (merge_switch scrutinee arms).
Proof.
  destruct scrutinee as [v | sym | c]; try reflexivity.
  destruct arms as [ | [i ac] [ | [i' ac'] others]]; try reflexivity.
  unfold merge_switch. cbn [map fst snd rename_simple]. cbv zeta.
  set (merged :=
    Apply_cont_expr (ac_continuation ac)
      (merged_arguments v (ac_args ac) (ac_args ac')) None).
  replace
    (Apply_cont_expr (ac_continuation (rename_apply_cont r ac))
       (merged_arguments (rename_variable r v)
          (ac_args (rename_apply_cont r ac))
          (ac_args (rename_apply_cont r ac')))
       None)
    with (rename_apply_cont r merged)
    by (unfold merged, rename_apply_cont;
        cbn [ac_continuation ac_args ac_trap_action option_map];
        rewrite merged_arguments_renamed; reflexivity).
  change (ac_args (rename_apply_cont r merged))
    with (map (rename_simple r) (ac_args merged)).
  rewrite var_in_simples_renamed.
  change
    ((i, rename_apply_cont r ac) :: (i', rename_apply_cont r ac')
     :: map (fun arm => (fst arm, rename_apply_cont r (snd arm))) others)
    with
      (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
         ((i, ac) :: (i', ac') :: others)).
  rewrite
    !(forallb_under_map _ _ (arm_is_merged _ v merged))
    by (intros arm; apply arm_is_merged_renamed).
  destruct (var_in_simples v (ac_args merged)); [ | reflexivity ].
  destruct (forallb (arm_is_merged By_tagging v merged) _); [ reflexivity | ].
  destruct (forallb (arm_is_merged By_tagging_and_negating v merged) _);
    [ reflexivity | ].
  destruct (forallb (arm_is_merged As_it_is v merged) _); [ reflexivity | ].
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  change (ac_args (rename_apply_cont r ac'))
    with (map (rename_simple r) (ac_args ac')).
  rewrite arithmetic_of_arms_renamed.
  destruct (arithmetic_of_arms i i' (ac_args ac) (ac_args ac'))
    as [[[kind offset] slope] | ]; [ | reflexivity ].
  rewrite length_map,
    (forallb_under_map _ _ (arm_is_merged _ v merged))
    by (intros arm; apply arm_is_merged_renamed).
  destruct (_ && _); [ | reflexivity ].
  rewrite bind_arithmetic_renamed. reflexivity.
Qed.

(* The ways chosen for each argument. A way, a row, a column and the
   continuations around, renamed. *)
Definition rename_way (r : renaming) (way : argument_way) : argument_way :=
  match way with
  | Same_in_every_arm s => Same_in_every_arm (rename_simple r s)
  | Computed how => Computed how
  end.

Definition rename_row (r : renaming) (row : Z * list simple)
    : Z * list simple :=
  (fst row, map (rename_simple r) (snd row)).

Definition rename_entry (r : renaming) (entry : Z * simple) : Z * simple :=
  (fst entry, rename_simple r (snd entry)).

Definition rename_known_continuations (r : renaming)
    (known : known_continuations) : known_continuations :=
  map
    (fun entry =>
       (rename_continuation r (fst entry),
        map (rename_variable r) (snd entry)))
    known.

Lemma among_names_renamed (r : renaming) (x : variable)
    (names : list variable) :
  among (rename_variable r x) (map (rename_variable r) names) = among x names.
Proof.
  unfold among. induction names as [ | y names IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb, IH. reflexivity.
Qed.

Lemma distinct_names_renamed (r : renaming) (names : list variable) :
  distinct (map (rename_variable r) names) = distinct names.
Proof.
  induction names as [ | y names IH]; [ reflexivity | ].
  cbn [map distinct]. rewrite among_names_renamed, IH. reflexivity.
Qed.

Lemma names_in_simple_renamed (r : renaming) (s : simple)
    (names : list variable) :
  existsb (fun x => var_in_simple x (rename_simple r s))
    (map (rename_variable r) names)
  = existsb (fun x => var_in_simple x s) names.
Proof.
  induction names as [ | y names IH]; simpl; [ reflexivity | ].
  rewrite IH. f_equal. destruct s; simpl; try reflexivity.
  apply rename_variable_eqb.
Qed.

Lemma argument_in_arm_renamed (r : renaming) (i : Z) (way : argument_way) :
  argument_in_arm i (rename_way r way)
  = option_map (rename_simple r) (argument_in_arm i way).
Proof.
  destruct way as [s | how]; simpl; [ reflexivity | ].
  destruct (constant_of_arm how i); reflexivity.
Qed.

Lemma arguments_in_arm_renamed (r : renaming) (i : Z)
    (ways : list argument_way) :
  arguments_in_arm i (map (rename_way r) ways)
  = option_map (map (rename_simple r)) (arguments_in_arm i ways).
Proof.
  induction ways as [ | way ways IH]; [ reflexivity | ].
  cbn [map arguments_in_arm]. rewrite argument_in_arm_renamed, IH.
  destruct (argument_in_arm i way); [ | reflexivity ].
  destruct (arguments_in_arm i ways); reflexivity.
Qed.

Lemma arm_has_ways_renamed (r : renaming) (k : continuation)
    (ways : list argument_way) (arm : Z * apply_cont) :
  arm_has_ways (rename_continuation r k) (map (rename_way r) ways)
    (fst arm, rename_apply_cont r (snd arm))
  = arm_has_ways k ways arm.
Proof.
  unfold arm_has_ways. cbn [fst snd]. rewrite arguments_in_arm_renamed.
  destruct (arguments_in_arm (fst arm) ways) as [args | ]; [ | reflexivity ].
  cbn [option_map].
  change
    (Apply_cont_expr (rename_continuation r k) (map (rename_simple r) args)
       None)
    with (rename_apply_cont r (Apply_cont_expr k args None)).
  destruct (apply_cont_eq_dec (snd arm) (Apply_cont_expr k args None))
    as [-> | Hne],
    (apply_cont_eq_dec (rename_apply_cont r _) (rename_apply_cont r _))
      as [Heq | Hne'];
    try reflexivity; [ contradiction | ].
  elim Hne. exact (renamed_apply_cont_same r _ _ Heq).
Qed.

Lemma column_of_arms_renamed (r : renaming) (rows : list (Z * list simple)) :
  column_of_arms (map (rename_row r) rows)
  = option_map (map (rename_entry r)) (column_of_arms rows).
Proof.
  induction rows as [ | [i [ | s args]] rows IH]; try reflexivity.
  cbn [map rename_row fst snd column_of_arms]. rewrite IH.
  destruct (column_of_arms rows); reflexivity.
Qed.

Lemma rest_of_arms_renamed (r : renaming) (rows : list (Z * list simple)) :
  rest_of_arms (map (rename_row r) rows)
  = map (rename_row r) (rest_of_arms rows).
Proof.
  unfold rest_of_arms. rewrite !map_map. apply map_ext.
  intros [i [ | s args]]; reflexivity.
Qed.

Lemma simple_is_renamed (r : renaming) (s s' : simple) :
  (if simple_eq_dec (rename_simple r s) (rename_simple r s')
   then true
   else false)
  = (if simple_eq_dec s s' then true else false).
Proof.
  destruct (simple_eq_dec s s') as [-> | Hne],
    (simple_eq_dec (rename_simple r _) (rename_simple r _)) as [Heq | Hne'];
    try reflexivity; [ contradiction | ].
  elim Hne. exact (renamed_simple_same r _ _ Heq).
Qed.

Lemma column_is_computed_renamed (r : renaming) (how : merge_by)
    (column : list (Z * simple)) :
  column_is_computed how (map (rename_entry r) column)
  = column_is_computed how column.
Proof.
  unfold column_is_computed. apply forallb_under_map. intros entry.
  cbn [rename_entry fst snd].
  destruct (constant_of_arm how (fst entry)) as [c | ]; [ | reflexivity ].
  exact (simple_is_renamed r (snd entry) (Const c)).
Qed.

Lemma way_of_column_renamed (r : renaming) (column : list (Z * simple)) :
  way_of_column (map (rename_entry r) column)
  = option_map (rename_way r) (way_of_column column).
Proof.
  destruct column as [ | [i s] [ | [i' s'] rest]]; try reflexivity.
  set (column := (i, s) :: (i', s') :: rest).
  assert
    (Hsame :
      forallb
        (fun entry =>
           if simple_eq_dec (rename_simple r s) (snd entry)
           then true
           else false)
        (map (rename_entry r) column)
      = forallb
          (fun entry => if simple_eq_dec s (snd entry) then true else false)
          column).
  { apply forallb_under_map. intros entry. cbn [rename_entry snd].
    apply simple_is_renamed. }
  change
    (way_of_column (map (rename_entry r) column)
     = option_map (rename_way r) (way_of_column column))
    with
      ((if forallb
             (fun entry =>
                if simple_eq_dec (rename_simple r s) (snd entry)
                then true
                else false)
             (map (rename_entry r) column)
        then Some (Same_in_every_arm (rename_simple r s))
        else if column_is_computed By_tagging (map (rename_entry r) column)
        then Some (Computed By_tagging)
        else if column_is_computed By_tagging_and_negating
                  (map (rename_entry r) column)
        then Some (Computed By_tagging_and_negating)
        else if column_is_computed As_it_is (map (rename_entry r) column)
        then Some (Computed As_it_is)
        else
          match rename_simple r s, rename_simple r s' with
          | Const c, Const c' =>
            match arithmetic_of_constants i i' c c' with
            | Some (kind, offset, slope) =>
              if (3 <=? length (map (rename_entry r) column))%nat
                 && column_is_computed (By_arithmetic kind offset slope)
                      (map (rename_entry r) column)
              then Some (Computed (By_arithmetic kind offset slope))
              else None
            | None => None
            end
          | _, _ => None
          end)
       = option_map (rename_way r)
           (if forallb
                 (fun entry =>
                    if simple_eq_dec s (snd entry) then true else false)
                 column
            then Some (Same_in_every_arm s)
            else if column_is_computed By_tagging column
            then Some (Computed By_tagging)
            else if column_is_computed By_tagging_and_negating column
            then Some (Computed By_tagging_and_negating)
            else if column_is_computed As_it_is column
            then Some (Computed As_it_is)
            else
              match s, s' with
              | Const c, Const c' =>
                match arithmetic_of_constants i i' c c' with
                | Some (kind, offset, slope) =>
                  if (3 <=? length column)%nat
                     && column_is_computed (By_arithmetic kind offset slope)
                          column
                  then Some (Computed (By_arithmetic kind offset slope))
                  else None
                | None => None
                end
              | _, _ => None
              end)).
  rewrite Hsame, !column_is_computed_renamed, length_map.
  destruct (forallb _ column); [ reflexivity | ].
  destruct (column_is_computed By_tagging column); [ reflexivity | ].
  destruct (column_is_computed By_tagging_and_negating column);
    [ reflexivity | ].
  destruct (column_is_computed As_it_is column); [ reflexivity | ].
  destruct s as [x | sym | c], s' as [x' | sym' | c']; try reflexivity.
  cbn [rename_simple].
  destruct (arithmetic_of_constants i i' c c') as [[[kind offset] slope] | ];
    [ | reflexivity ].
  rewrite column_is_computed_renamed.
  destruct (_ && _); reflexivity.
Qed.

Lemma ways_of_arguments_renamed (r : renaming) (count : nat) :
  ∀ rows : list (Z * list simple),
    ways_of_arguments count (map (rename_row r) rows)
    = option_map (map (rename_way r)) (ways_of_arguments count rows).
Proof.
  induction count as [ | count IH]; intros rows; [ reflexivity | ].
  cbn [ways_of_arguments]. rewrite column_of_arms_renamed.
  destruct (column_of_arms rows) as [column | ]; [ | reflexivity ].
  cbn [option_map]. rewrite way_of_column_renamed.
  destruct (way_of_column column) as [way | ]; [ | reflexivity ].
  cbn [option_map]. rewrite rest_of_arms_renamed, IH.
  destruct (ways_of_arguments count (rest_of_arms rows)); reflexivity.
Qed.

Lemma bind_way_renamed (r : renaming) (how : merge_by) (v x : variable)
    (body : expr) :
  rename_expr r (bind_way how v x body)
  = bind_way how (rename_variable r v) (rename_variable r x)
      (rename_expr r body).
Proof.
  destruct how as [ | | | [ | ] offset slope]; cbn [bind_way]; cbv zeta;
    unfold bind_tagged_to, bind_same_to, bind_negated;
    cbn [rename_expr rename_named rename_simple map];
    rewrite ?bind_arithmetic_steps_renamed; reflexivity.
Qed.

Lemma bind_ways_renamed (r : renaming) (v : variable) (body : expr)
    (names : list variable) :
  ∀ ways : list argument_way,
    rename_expr r (bind_ways v names ways body)
    = bind_ways (rename_variable r v) (map (rename_variable r) names)
        (map (rename_way r) ways) (rename_expr r body).
Proof.
  induction names as [ | x names IH]; intros [ | [s | how] ways];
    cbn [bind_ways map rename_way]; try reflexivity; [ apply IH | ].
  rewrite bind_way_renamed, IH. reflexivity.
Qed.

Lemma merged_by_ways_renamed (r : renaming) (v : variable)
    (names : list variable) :
  ∀ ways : list argument_way,
    map (rename_simple r) (merged_by_ways v names ways)
    = merged_by_ways (rename_variable r v) (map (rename_variable r) names)
        (map (rename_way r) ways).
Proof.
  induction names as [ | x names IH]; intros [ | [s | how] ways];
    cbn [merged_by_ways map rename_way]; try reflexivity;
    rewrite IH; [ reflexivity | ].
  destruct how; reflexivity.
Qed.

Lemma names_usable_renamed (r : renaming) (v : variable)
    (names : list variable) (ways : list argument_way) :
  names_usable (rename_variable r v) (map (rename_variable r) names)
    (map (rename_way r) ways)
  = names_usable v names ways.
Proof.
  unfold names_usable.
  rewrite !length_map, distinct_names_renamed, among_names_renamed.
  f_equal. apply forallb_under_map. intros [s | how]; [ | reflexivity ].
  cbn [rename_way]. rewrite names_in_simple_renamed. reflexivity.
Qed.

Lemma merge_switch_by_arguments_renamed (r : renaming)
    (names : list variable) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  merge_switch_by_arguments (map (rename_variable r) names)
    (rename_simple r scrutinee)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = option_map (rename_expr r)
      (merge_switch_by_arguments names scrutinee arms).
Proof.
  destruct scrutinee as [v | sym | c]; try reflexivity.
  destruct arms as [ | [i ac] [ | arm1 others]]; try reflexivity.
  set (arms := (i, ac) :: arm1 :: others).
  assert
    (Hrows :
      map (fun arm => (fst arm, ac_args (snd arm)))
        (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
      = map (rename_row r)
          (map (fun arm => (fst arm, ac_args (snd arm))) arms)).
  { rewrite !map_map. apply map_ext. intros arm. reflexivity. }
  change
    (merge_switch_by_arguments (map (rename_variable r) names)
       (rename_simple r (Var v))
       (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms))
    with
      (let* ways :=
         ways_of_arguments (length (map (rename_simple r) (ac_args ac)))
           (map (fun arm => (fst arm, ac_args (snd arm)))
              (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                 arms))
       in
       if names_usable (rename_variable r v) (map (rename_variable r) names)
            ways
          && forallb
               (arm_has_ways (rename_continuation r (ac_continuation ac))
                  ways)
               (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                  arms)
       then
         Some
           (bind_ways (rename_variable r v) (map (rename_variable r) names)
              ways
              (Apply_cont
                 (Apply_cont_expr (rename_continuation r (ac_continuation ac))
                    (merged_by_ways (rename_variable r v)
                       (map (rename_variable r) names) ways)
                    None)))
       else None).
  change (merge_switch_by_arguments names (Var v) arms)
    with
      (let* ways :=
         ways_of_arguments (length (ac_args ac))
           (map (fun arm => (fst arm, ac_args (snd arm))) arms)
       in
       if names_usable v names ways
          && forallb (arm_has_ways (ac_continuation ac) ways) arms
       then
         Some
           (bind_ways v names ways
              (Apply_cont
                 (Apply_cont_expr (ac_continuation ac)
                    (merged_by_ways v names ways) None)))
       else None).
  rewrite Hrows, length_map, ways_of_arguments_renamed.
  destruct (ways_of_arguments _ _) as [ways | ]; [ | reflexivity ].
  cbn [option_map].
  rewrite names_usable_renamed,
    (forallb_under_map _ _ (arm_has_ways (ac_continuation ac) ways))
    by (intros arm; apply arm_has_ways_renamed).
  destruct (_ && _); [ | reflexivity ].
  cbn [option_map]. rewrite bind_ways_renamed.
  cbn [rename_expr]. unfold rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action option_map].
  rewrite merged_by_ways_renamed. reflexivity.
Qed.

Lemma lookup_known_renamed (r : renaming) (k : continuation)
    (known : known_continuations) :
  lookup (rename_continuation r k) (rename_known_continuations r known)
  = option_map (map (rename_variable r)) (lookup k known).
Proof.
  induction known as [ | [k' names] known IH]; [ reflexivity | ].
  cbn [rename_known_continuations map fst snd lookup].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma merge_switch_knowing_renamed (r : renaming)
    (known : known_continuations) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  merge_switch_knowing (rename_known_continuations r known)
    (rename_simple r scrutinee)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = rename_expr r (merge_switch_knowing known scrutinee arms).
Proof.
  unfold merge_switch_knowing. rewrite merge_switch_equivariant.
  destruct (merge_switch scrutinee arms); try reflexivity.
  cbn [rename_expr].
  destruct arms as [ | [i ac] others]; [ reflexivity | ].
  set (arms := (i, ac) :: others).
  change
    (match
       map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms
     with
     | [] =>
       Switch (rename_simple r scrutinee)
         (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
     | (_, ac0) :: _ =>
       match
         lookup (ac_continuation ac0) (rename_known_continuations r known)
       with
       | Some names =>
         match
           merge_switch_by_arguments names (rename_simple r scrutinee)
             (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
         with
         | Some merged => merged
         | None =>
           Switch (rename_simple r scrutinee)
             (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
         end
       | None =>
         Switch (rename_simple r scrutinee)
           (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
       end
     end)
    with
      (match
         lookup (rename_continuation r (ac_continuation ac))
           (rename_known_continuations r known)
       with
       | Some names =>
         match
           merge_switch_by_arguments names (rename_simple r scrutinee)
             (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
         with
         | Some merged => merged
         | None =>
           Switch (rename_simple r scrutinee)
             (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
         end
       | None =>
         Switch (rename_simple r scrutinee)
           (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
       end).
  rewrite lookup_known_renamed.
  destruct (lookup (ac_continuation ac) known) as [names | ];
    [ | reflexivity ].
  cbn [option_map]. rewrite merge_switch_by_arguments_renamed.
  destruct (merge_switch_by_arguments names scrutinee arms); reflexivity.
Qed.

Lemma known_of_handlers_renamed (r : renaming)
    (handlers : list (continuation * continuation_handler))
    (known : known_continuations) :
  map (fun entry => (fst entry, parameters_of_handler (snd entry)))
    (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
       handlers)
  ++ rename_known_continuations r known
  = rename_known_continuations r
      (map (fun entry => (fst entry, parameters_of_handler (snd entry)))
         handlers
       ++ known).
Proof.
  unfold rename_known_continuations. rewrite map_app, !map_map. f_equal.
  apply map_ext. intros [k [params metadata handler]]. reflexivity.
Qed.

Fixpoint merge_switches_knowing_equivariant (r : renaming)
    (known : known_continuations) (e : expr) {struct e} :
  merge_switches_knowing (rename_known_continuations r known) (rename_expr r e)
  = rename_expr r (merge_switches_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite merge_switches_knowing_equivariant. reflexivity.
  - rewrite merge_switches_knowing_equivariant. reflexivity.
  - change
      ((rename_continuation r k, map (rename_variable r) params)
       :: rename_known_continuations r known)
      with (rename_known_continuations r ((k, params) :: known)).
    rewrite !merge_switches_knowing_equivariant. reflexivity.
  - rewrite known_of_handlers_renamed, merge_switches_knowing_equivariant.
    reflexivity.
  - apply merge_switch_knowing_renamed.
Qed.

Theorem merge_switches_equivariant : merge_switches_equivariant_statement.
Proof.
  intros r e. exact (merge_switches_knowing_equivariant r [] e).
Qed.

(* ======================================================================== *)
(* Five terms                                                               *)
(* ======================================================================== *)

Theorem merge_switches_tagging_example :
  merge_switches_tagging_example_statement.
Proof. reflexivity. Qed.

Theorem merge_switches_negating_example :
  merge_switches_negating_example_statement.
Proof. reflexivity. Qed.

Theorem merge_switches_as_it_is_example :
  merge_switches_as_it_is_example_statement.
Proof. reflexivity. Qed.

Theorem merge_switches_arithmetic_example :
  merge_switches_arithmetic_example_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem merge_switches_two_ways_example :
  merge_switches_two_ways_example_statement.
Proof. vm_compute. reflexivity. Qed.
