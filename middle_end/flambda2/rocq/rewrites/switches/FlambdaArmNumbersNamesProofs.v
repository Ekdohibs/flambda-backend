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


(* FlambdaArmNumbersSpec.numbers_in_arms commutes with renaming and
   keeps scope: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and FlambdaStaticsScopeProofs.v
   prove the two statements. There is no statement here, and nothing
   here needs review.

   The pass compares names by equality only and makes none, so each of
   its functions gives, on a renamed term, the renamed result; the
   variables that it takes in turn are those of the renamed term in
   the same order, and the variable that one of them is the
   untagging of is renamed with it. For the scope: the rewrite puts a
   constant for a variable, so a free variable of the result is one of
   the term; the continuations and the code are as they were. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaUnaliasParamsNamesProofs FlambdaUntagParamsSpec
  FlambdaUntagParamsNamesProofs FlambdaGiveScrutineeSpec
  FlambdaGiveScrutineeNamesProofs FlambdaArmNumbersSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

Lemma number_simple_equivariant (r : renaming) (x : variable) (c : Z)
    (s : simple) :
  number_simple (rename_variable r x) c (rename_simple r s)
  = rename_simple r (number_simple x c s).
Proof.
  destruct s as [y | sym | c0]; cbn [rename_simple number_simple];
    try reflexivity.
  rewrite rename_variable_eqb. destruct (Pos.eqb y x); reflexivity.
Qed.

Lemma number_in_apply_cont_equivariant (r : renaming) (x : variable)
    (known : option Z) (ac : apply_cont) :
  number_in_apply_cont (rename_variable r x) known (rename_apply_cont r ac)
  = rename_apply_cont r (number_in_apply_cont x known ac).
Proof.
  unfold number_in_apply_cont. destruct known as [c | ]; [ | reflexivity ].
  unfold rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite !map_map. f_equal. apply map_ext. intros s.
  apply number_simple_equivariant.
Qed.

(* The variable that [x] is the untagging of is renamed with the
   term. *)
Lemma source_under_let_equivariant (r : renaming) (x y : variable)
    (d : named) (source : option variable) :
  source_under_let (rename_variable r x) (rename_variable r y)
    (rename_named r d) (option_map (rename_variable r) source)
  = option_map (rename_variable r) (source_under_let x y d source).
Proof.
  unfold source_under_let. rewrite rename_variable_eqb, untag_of_renamed.
  destruct (Pos.eqb y x).
  - destruct (untag_of d) as [t | ]; cbn [option_map]; [ | reflexivity ].
    rewrite rename_variable_eqb. destruct (Pos.eqb t x); reflexivity.
  - destruct source as [t | ]; cbn [option_map]; [ | reflexivity ].
    rewrite rename_variable_eqb. destruct (Pos.eqb y t); reflexivity.
Qed.

Lemma source_under_equivariant (r : renaming) (x : variable)
    (params : list variable) (source : option variable) :
  source_under (rename_variable r x) (map (rename_variable r) params)
    (option_map (rename_variable r) source)
  = option_map (rename_variable r) (source_under x params source).
Proof.
  unfold source_under. rewrite among_rename_variable.
  destruct (among x params); [ reflexivity | ].
  destruct source as [t | ]; cbn [option_map]; [ | reflexivity ].
  rewrite among_rename_variable. destruct (among t params); reflexivity.
Qed.

Lemma number_read_equivariant (r : renaming) (x : variable)
    (source : option variable) (c : Z) (s : simple) :
  number_read (rename_variable r x) (option_map (rename_variable r) source) c
    (rename_simple r s)
  = rename_simple r (number_read x source c s).
Proof.
  destruct s as [y | sym | c0]; cbn [rename_simple number_read];
    try reflexivity.
  rewrite rename_variable_eqb. destruct (Pos.eqb y x); [ reflexivity | ].
  destruct source as [t | ]; cbn [option_map]; [ | reflexivity ].
  rewrite rename_variable_eqb. destruct (Pos.eqb y t); reflexivity.
Qed.

Lemma number_in_named_equivariant (r : renaming) (x : variable)
    (source : option variable) (known : option Z) (d : named) :
  number_in_named (rename_variable r x)
    (option_map (rename_variable r) source) known (rename_named r d)
  = rename_named r (number_in_named x source known d).
Proof.
  unfold number_in_named. destruct known as [c | ]; [ | reflexivity ].
  destruct d as [s | p args | | | | | ]; cbn [rename_named];
    try reflexivity.
  - rewrite number_read_equivariant. reflexivity.
  - rewrite !map_map. f_equal. apply map_ext. intros s.
    apply number_read_equivariant.
Qed.

Lemma number_in_scrutinee_equivariant (r : renaming) (x : variable)
    (source : option variable) (known : option Z) (scrutinee : simple) :
  number_in_scrutinee (rename_variable r x)
    (option_map (rename_variable r) source) known (rename_simple r scrutinee)
  = rename_simple r (number_in_scrutinee x source known scrutinee).
Proof.
  unfold number_in_scrutinee. destruct known as [c | ]; [ | reflexivity ].
  apply number_read_equivariant.
Qed.

Fixpoint number_in_region_equivariant (r : renaming) (x : variable)
    (source : option variable) (known : option Z) (e : expr) {struct e} :
  number_in_region (rename_variable r x)
    (option_map (rename_variable r) source) known (rename_expr r e)
  = rename_expr r (number_in_region x source known e)
with number_in_handler_equivariant (r : renaming) (x : variable)
    (source : option variable) (known : option Z) (h : continuation_handler)
    {struct h} :
  number_in_handler (rename_variable r x)
    (option_map (rename_variable r) source) known (rename_handler r h)
  = rename_handler r (number_in_handler x source known h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr number_in_region]; try reflexivity.
    + rewrite rename_variable_eqb, number_in_named_equivariant,
        source_under_let_equivariant, number_in_region_equivariant.
      reflexivity.
    + rewrite number_in_region_equivariant. reflexivity.
    + rewrite arm_fact_equivariant, number_in_handler_equivariant,
        number_in_region_equivariant.
      reflexivity.
    + rewrite number_in_region_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map]. rewrite number_in_handler_equivariant, IH. reflexivity.
    + rewrite number_in_apply_cont_equivariant. reflexivity.
    + rewrite number_in_scrutinee_equivariant. f_equal.
      induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
      cbn [map fst snd].
      rewrite arm_known_equivariant, number_in_apply_cont_equivariant, IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler number_in_handler].
    rewrite known_under_equivariant, source_under_equivariant,
      number_in_region_equivariant.
    reflexivity.
Qed.

Lemma number_in_arms_equivariant (r : renaming) (x : variable)
    (known : option Z) (e : expr) :
  number_in_arms (rename_variable r x) known (rename_expr r e)
  = rename_expr r (number_in_arms x known e).
Proof. exact (number_in_region_equivariant r x None known e). Qed.

Fixpoint scrutinees_equivariant (r : renaming) (e : expr) {struct e} :
  scrutinees (rename_expr r e) = map (rename_variable r) (scrutinees e)
with handler_scrutinees_equivariant (r : renaming)
    (h : continuation_handler) {struct h} :
  handler_scrutinees (rename_handler r h)
  = map (rename_variable r) (handler_scrutinees h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr scrutinees]; try reflexivity.
    + apply scrutinees_equivariant.
    + apply scrutinees_equivariant.
    + rewrite map_app, handler_scrutinees_equivariant,
        scrutinees_equivariant.
      reflexivity.
    + rewrite map_app, scrutinees_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map flat_map].
      rewrite map_app, handler_scrutinees_equivariant, IH. reflexivity.
    + destruct scrutinee as [y | sym | cst]; reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_scrutinees].
    apply scrutinees_equivariant.
Qed.

Lemma numbers_fold_equivariant (r : renaming) (xs : list variable) :
  ∀ e : expr,
    fold_left (fun e x => number_in_arms x None e) (rename_expr r e)
      (map (rename_variable r) xs)
    = rename_expr r (fold_left (fun e x => number_in_arms x None e) e xs).
Proof.
  induction xs as [ | x xs IH]; intros e; unfold fold_left;
    cbn [map List.fold_left]; [ reflexivity | ].
  rewrite number_in_arms_equivariant. exact (IH (number_in_arms x None e)).
Qed.

Lemma numbers_in_arms_renamed : rewrite_equivariant numbers_in_arms.
Proof.
  intros r e. unfold numbers_in_arms.
  rewrite scrutinees_equivariant. apply numbers_fold_equivariant.
Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Lemma number_in_apply_cont_continuations (x : variable) (known : option Z)
    (ac : apply_cont) :
  apply_cont_continuations (number_in_apply_cont x known ac)
  = apply_cont_continuations ac.
Proof. unfold number_in_apply_cont. destruct known; reflexivity. Qed.

Lemma number_simples_free_variables (x : variable) (c : Z) (a : variable)
    (args : list simple) :
  In a (free_variables_of_simples (map (number_simple x c) args)) ->
  In a (free_variables_of_simples args).
Proof.
  unfold free_variables_of_simples.
  induction args as [ | s args IH]; cbn [map flat_map];
    [ intros Hin; exact Hin | ].
  intros Hin. apply in_app_or in Hin as [Hin | Hin].
  - apply in_or_app. left.
    destruct s as [y | sym | c0]; cbn [number_simple] in Hin;
      try exact Hin.
    destruct (Pos.eqb y x); [ contradiction | exact Hin ].
  - apply in_or_app. right. exact (IH Hin).
Qed.

Lemma number_in_apply_cont_free_variables (x : variable) (known : option Z)
    (ac : apply_cont) (a : variable) :
  In a (free_variables_of_apply_cont (number_in_apply_cont x known ac)) ->
  In a (free_variables_of_apply_cont ac).
Proof.
  unfold number_in_apply_cont, free_variables_of_apply_cont.
  destruct known as [c | ]; [ | intros Hin; exact Hin ].
  cbn [ac_args]. apply number_simples_free_variables.
Qed.

Lemma number_read_free_variables (x : variable) (source : option variable)
    (c : Z) (a : variable) (s : simple) :
  In a (free_variables_of_simple (number_read x source c s)) ->
  In a (free_variables_of_simple s).
Proof.
  destruct s as [y | sym | c0]; cbn [number_read];
    try (intros Hin; exact Hin).
  destruct (Pos.eqb y x); [ intros [] | ].
  destruct source as [t | ]; [ | intros Hin; exact Hin ].
  destruct (Pos.eqb y t); [ intros [] | intros Hin; exact Hin ].
Qed.

Lemma number_reads_free_variables (x : variable) (source : option variable)
    (c : Z) (a : variable) (args : list simple) :
  In a (free_variables_of_simples (map (number_read x source c) args)) ->
  In a (free_variables_of_simples args).
Proof.
  unfold free_variables_of_simples.
  induction args as [ | s args IH]; cbn [map flat_map];
    [ intros Hin; exact Hin | ].
  intros Hin. apply in_app_or in Hin as [Hin | Hin]; apply in_or_app.
  - left. exact (number_read_free_variables x source c a s Hin).
  - right. exact (IH Hin).
Qed.

Lemma number_in_named_free_variables (x : variable)
    (source : option variable) (known : option Z) (a : variable) (d : named) :
  In a (free_variables_of_named (number_in_named x source known d)) ->
  In a (free_variables_of_named d).
Proof.
  unfold number_in_named. destruct known as [c | ]; [ | intros Hin; exact Hin ].
  destruct d as [s | p args | | | | | ]; cbn [free_variables_of_named];
    try (intros Hin; exact Hin).
  - apply number_read_free_variables.
  - apply number_reads_free_variables.
Qed.

Lemma number_in_scrutinee_free_variables (x : variable)
    (source : option variable) (known : option Z) (a : variable)
    (scrutinee : simple) :
  In a
    (free_variables_of_simple (number_in_scrutinee x source known scrutinee)) ->
  In a (free_variables_of_simple scrutinee).
Proof.
  unfold number_in_scrutinee.
  destruct known as [c | ]; [ | intros Hin; exact Hin ].
  apply number_read_free_variables.
Qed.

Fixpoint number_in_region_code_closed (x : variable)
    (source : option variable) (known : option Z) (e : expr) {struct e} :
  code_closed (number_in_region x source known e) = code_closed e
with number_in_handler_code_closed (x : variable)
    (source : option variable) (known : option Z) (h : continuation_handler)
    {struct h} :
  handler_code_closed (number_in_handler x source known h)
  = handler_code_closed h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [number_in_region code_closed]; try reflexivity.
    + apply number_in_region_code_closed.
    + rewrite number_in_region_code_closed. reflexivity.
    + rewrite number_in_handler_code_closed, number_in_region_code_closed.
      reflexivity.
    + rewrite number_in_region_code_closed. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite number_in_handler_code_closed, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [number_in_handler handler_code_closed].
    apply number_in_region_code_closed.
Qed.

Fixpoint number_in_region_free_continuations (x : variable)
    (source : option variable) (known : option Z) (e : expr) {struct e} :
  free_continuations_of_expr (number_in_region x source known e)
  = free_continuations_of_expr e
with number_in_handler_free_continuations (x : variable)
    (source : option variable) (known : option Z) (h : continuation_handler)
    {struct h} :
  free_continuations_of_handler (number_in_handler x source known h)
  = free_continuations_of_handler h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [number_in_region free_continuations_of_expr]; try reflexivity.
    + apply number_in_region_free_continuations.
    + rewrite number_in_region_free_continuations. reflexivity.
    + rewrite number_in_handler_free_continuations,
        number_in_region_free_continuations.
      reflexivity.
    + rewrite number_in_region_free_continuations. f_equal.
      * induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map fst]. rewrite IH. reflexivity.
      * f_equal.
        induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map flat_map].
        rewrite number_in_handler_free_continuations, IH. reflexivity.
    + apply number_in_apply_cont_continuations.
    + induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
      cbn [map flat_map snd].
      rewrite number_in_apply_cont_continuations, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [number_in_handler free_continuations_of_handler].
    apply number_in_region_free_continuations.
Qed.

(* A free variable of the result is one of the term. *)
Fixpoint number_in_region_free_variables (x a : variable) (e : expr)
    {struct e} :
  ∀ (source : option variable) (known : option Z),
    In a (free_variables_of_expr (number_in_region x source known e)) ->
    In a (free_variables_of_expr e)
with number_in_handler_free_variables (x a : variable)
    (h : continuation_handler) {struct h} :
  ∀ (source : option variable) (known : option Z),
    In a (free_variables_of_handler (number_in_handler x source known h)) ->
    In a (free_variables_of_handler h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros source known; cbn [number_in_region free_variables_of_expr];
      try (intros Hin; exact Hin).
    + intros Hin. apply in_app_or in Hin as [Hin | Hin].
      * apply in_or_app. left.
        exact (number_in_named_free_variables x source known a d Hin).
      * apply in_without in Hin as [Hin Hy].
        apply in_or_app. right. apply in_without. split; [ | exact Hy ].
        exact (number_in_region_free_variables x a body _ _ Hin).
    + intros Hin. apply in_app_or in Hin as [Hin | Hin];
        [ apply in_or_app; left; exact Hin | ].
      apply in_or_app. right.
      exact (number_in_region_free_variables x a body source known Hin).
    + intros Hin. apply in_app_or in Hin as [Hin | Hin]; apply in_or_app.
      * left. exact (number_in_handler_free_variables x a hh _ _ Hin).
      * right.
        exact (number_in_region_free_variables x a body source known Hin).
    + intros Hin. apply in_app_or in Hin as [Hin | Hin]; apply in_or_app.
      * left.
        induction handlers as [ | [k' hh] handlers IH];
          cbn [map flat_map] in Hin |- *; [ contradiction | ].
        apply in_app_or in Hin as [Hin | Hin]; apply in_or_app.
        -- left.
           exact (number_in_handler_free_variables x a hh source known Hin).
        -- right. exact (IH Hin).
      * right.
        exact (number_in_region_free_variables x a body source known Hin).
    + apply number_in_apply_cont_free_variables.
    + intros Hin. apply in_app_or in Hin as [Hin | Hin]; apply in_or_app.
      * left.
        exact
          (number_in_scrutinee_free_variables x source known a scrutinee Hin).
      * right.
        induction arms as [ | [d ac] arms IH];
          cbn [map flat_map snd] in Hin |- *; [ contradiction | ].
        apply in_app_or in Hin as [Hin | Hin]; apply in_or_app.
        -- left. exact (number_in_apply_cont_free_variables x _ ac a Hin).
        -- right. exact (IH Hin).
  - destruct h as [params metadata handler]. intros source known.
    cbn [number_in_handler free_variables_of_handler]. intros Hin.
    apply in_without in Hin as [Hin Hparams].
    apply in_without. split; [ | exact Hparams ].
    exact (number_in_region_free_variables x a handler _ _ Hin).
Qed.

Lemma number_in_arms_scoped (x : variable) (e : expr) :
  scoped e (number_in_arms x None e).
Proof.
  intros Hclosed. unfold number_in_arms.
  rewrite number_in_region_code_closed.
  split; [ exact Hclosed | split ].
  - intros a Ha. exact (number_in_region_free_variables x a e None None Ha).
  - rewrite number_in_region_free_continuations. apply incl_refl.
Qed.

Lemma numbers_fold_scoped (xs : list variable) :
  ∀ e : expr,
    scoped e (fold_left (fun e x => number_in_arms x None e) e xs).
Proof.
  induction xs as [ | x xs IH]; intros e; unfold fold_left;
    cbn [List.fold_left]; [ apply scoped_refl | ].
  exact (scoped_trans _ _ _ (number_in_arms_scoped x e)
    (IH (number_in_arms x None e))).
Qed.

Lemma numbers_in_arms_scope : keeps_scope numbers_in_arms.
Proof.
  apply keeps_scope_intro. intros e. apply numbers_fold_scoped.
Qed.
