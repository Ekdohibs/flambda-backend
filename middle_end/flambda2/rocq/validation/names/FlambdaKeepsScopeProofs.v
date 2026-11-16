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

(* Proofs of the statements of FlambdaKeepsScopeSpec.v. Nothing here
   needs review. The relation between a term and one that keeps its
   scope, with its lemmas, is in FlambdaScopedProofs.v. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaBindersSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaFreeNamesSpec FlambdaFreeNamesProofs
  FlambdaApplySpec FlambdaCompareSpec FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaBlockLoadSpec FlambdaCseSpec FlambdaMadeLengthsSpec
  FlambdaIdentitiesSpec FlambdaMergeSwitchesSpec FlambdaRenameApartSpec
  FlambdaStaticsOrderSpec
  FlambdaUnboxSpec FlambdaUnusedContsSpec FlambdaUnusedContsProofs
  FlambdaUnusedStaticsSpec FlambdaValidateSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaKeepsScopeSpec.
From OxCamlFlambda2 Require FlambdaCseProofs.
From OxCamlFlambda2 Require Import FlambdaScopeTactic.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec FlambdaAliasesSpec.
Import ListNotations.

(* ======================================================================== *)
(* The comparison                                                           *)
(* ======================================================================== *)

Theorem expr_eqb_keeps_scope : expr_eqb_keeps_scope_statement.
Proof.
  intros e e' Hclosed Hsame.
  destruct (expr_eqb_scoped e e' Hsame Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | ].
  split; intros name Hname; apply among_In; [ apply Hv | apply Hk ];
    apply among_In; exact Hname.
Qed.

(* ======================================================================== *)
(* Every binder given a name of its own                                     *)
(* ======================================================================== *)

Theorem rename_apart_avoiding_keeps_scope :
  rename_apart_avoiding_keeps_scope_statement.
Proof.
  intros avoid e Hclosed. unfold rename_apart_avoiding. cbv zeta.
  destruct (expr_eqb [] [] e (canonical_avoiding avoid e)) eqn:Hsame.
  - exact
      (expr_eqb_keeps_scope e (canonical_avoiding avoid e) Hclosed Hsame).
  - split; [ exact Hclosed | split; intros name Hname; exact Hname ].
Qed.

Theorem rename_apart_keeps_scope : rename_apart_keeps_scope_statement.
Proof. exact (rename_apart_avoiding_keeps_scope []). Qed.

(* ======================================================================== *)
(* The rewrites that need to know nothing of where they are applied         *)
(* ======================================================================== *)

(* Folding constants. *)

Lemma fold_named_free (d : named) :
  incl (free_variables_of_named (fold_named d)) (free_variables_of_named d).
Proof.
  destruct d as [s | p args | | | | | ]; try apply incl_refl.
  destruct (fold_named_prim_cases p args)
    as [(kind & set_kind & -> & ->) | [(kind & -> & ->) | ->]];
    try apply incl_refl.
  destruct (if prim_may_fold p then consts_of_simples args else None)
    as [cs | ]; [ | apply incl_refl ].
  destruct (eval_prim empty_static_env p (map Value_const cs)) as [v | ];
    [ | apply incl_refl ].
  destruct v; try apply incl_refl. intros a [].
Qed.

Lemma fold_switch_scoped (scrutinee : simple) (arms : list (Z * apply_cont)) :
  scoped (Switch scrutinee arms) (fold_switch scrutinee arms).
Proof.
  unfold fold_switch. destruct scrutinee as [x | sym | c];
    try apply scoped_refl.
  destruct c; try apply scoped_refl.
  match goal with
  | |- context [find_arm ?i arms] =>
    destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ | apply scoped_refl ]
  end.
  apply find_arm_In in Hfind. destruct Hfind as [j Hin].
  exact (scoped_switch_arm _ _ j ac Hin).
Qed.

Fixpoint fold_constants_scoped (e : expr) : scoped e (fold_constants e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [fold_constants];
    first
      [ apply scoped_let;
        [ apply fold_named_free | apply fold_constants_scoped ]
      | apply fold_switch_scoped
      | scoped_under_binders fold_constants_scoped ].
Qed.

Theorem fold_constants_keeps_scope : fold_constants_keeps_scope_statement.
Proof. apply keeps_scope_intro, fold_constants_scoped. Qed.

(* Switches whose arms are all the same. *)

Lemma collapse_switch_scoped (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  scoped (Switch scrutinee arms) (collapse_switch scrutinee arms).
Proof.
  unfold collapse_switch. destruct arms as [ | [i ac] others];
    [ apply scoped_refl | ].
  destruct (forallb _ others); [ | apply scoped_refl ].
  exact
    (scoped_switch_arm scrutinee ((i, ac) :: others) i ac
       (or_introl eq_refl)).
Qed.

Fixpoint collapse_switches_scoped (e : expr) :
  scoped e (collapse_switches e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [collapse_switches];
    first
      [ apply collapse_switch_scoped
      | scoped_under_binders collapse_switches_scoped ].
Qed.

Theorem collapse_switches_keeps_scope :
  collapse_switches_keeps_scope_statement.
Proof. apply keeps_scope_intro, collapse_switches_scoped. Qed.

(* Switches whose arms are merged. The merged arguments are among
   those of the first arm and the variable of the scrutinee, and the
   continuation is that of the first arm; the new [Let]s, where there
   are any, bind that variable and use nothing else. *)

Lemma merged_arguments_free (v : variable) (args args' : list simple) :
  incl (free_variables_of_simples (merged_arguments v args args'))
    (v :: free_variables_of_simples args).
Proof.
  revert args'.
  induction args as [ | s args IH]; intros [ | s' args']; simpl;
    try (intros x []).
  assert
    (Hrest :
      incl (free_variables_of_simples (merged_arguments v args args'))
        (v :: free_variables_of_simple s ++ free_variables_of_simples args)).
  { intros x Hx. destruct (IH args' x Hx) as [<- | Hin];
      [ left; reflexivity | right; apply in_or_app; right; exact Hin ]. }
  destruct (simple_eq_dec s s'); simpl.
  - apply incl_app; [ apply incl_tl, incl_appl, incl_refl | exact Hrest ].
  - intros x [<- | Hx]; [ left; reflexivity | apply Hrest, Hx ].
Qed.

(* What the new [Let]s do to the scope of what is under them. *)
Definition binds_again (v : variable) (wrap : expr -> expr) : Prop :=
  ∀ body : expr,
    code_closed (wrap body) = code_closed body
    ∧ incl (free_variables_of_expr (wrap body))
        (v :: free_variables_of_expr body)
    ∧ free_continuations_of_expr (wrap body)
      = free_continuations_of_expr body.

Lemma bind_tagged_binds_again (v : variable) : binds_again v (bind_tagged v).
Proof.
  intros body. split; [ reflexivity | split; [ | reflexivity ] ].
  unfold bind_tagged. scope. simpl.
  intros x [<- | Hx]; [ left; reflexivity | right ].
  apply in_without in Hx. apply Hx.
Qed.

Lemma bind_negated_binds_again (v : variable) :
  binds_again v (bind_negated v).
Proof.
  intros body. split; [ reflexivity | split; [ | reflexivity ] ].
  unfold bind_negated. scope. simpl.
  intros x [<- | Hx]; [ left; reflexivity | right ].
  apply in_without in Hx. apply Hx.
Qed.

Lemma binds_again_nothing (v : variable) : binds_again v (fun body => body).
Proof.
  intros body. split; [ reflexivity | split; [ | reflexivity ] ].
  apply incl_tl, incl_refl.
Qed.

Lemma binds_again_twice (v : variable) (wrap wrap' : expr -> expr) :
  binds_again v wrap -> binds_again v wrap' ->
  binds_again v (fun body => wrap (wrap' body)).
Proof.
  intros Hwrap Hwrap' body.
  destruct (Hwrap (wrap' body)) as (Hclosed & Hv & Hk).
  destruct (Hwrap' body) as (Hclosed' & Hv' & Hk').
  split; [ congruence | split; [ | congruence ] ].
  intros x Hx. apply Hv in Hx.
  destruct Hx as [<- | Hx]; [ left; reflexivity | apply Hv', Hx ].
Qed.

(* The [Let]s of the arithmetic: each binds the variable again, to a
   primitive of itself and a constant. *)
Lemma bind_arithmetic_step_binds_again (standard : standard_int)
    (v : variable) (step : arithmetic_step) :
  binds_again v (bind_arithmetic_step standard v step).
Proof.
  intros body. split; [ reflexivity | split; [ | reflexivity ] ].
  unfold bind_arithmetic_step. destruct step; scope; simpl;
    (intros x [<- | Hx]; [ left; reflexivity | right ]);
    apply in_without in Hx; apply Hx.
Qed.

Lemma bind_arithmetic_steps_binds_again (standard : standard_int)
    (v : variable) (steps : list arithmetic_step) :
  binds_again v (bind_arithmetic_steps standard v steps).
Proof.
  induction steps as [ | step steps IH]; [ apply binds_again_nothing | ].
  exact
    (binds_again_twice v _ _
       (bind_arithmetic_step_binds_again standard v step) IH).
Qed.

Lemma bind_arithmetic_binds_again (kind : arithmetic_kind) (offset slope : Z)
    (v : variable) :
  binds_again v (bind_arithmetic kind offset slope v).
Proof.
  destruct kind.
  - exact
      (binds_again_twice v _ _ (bind_tagged_binds_again v)
         (bind_arithmetic_steps_binds_again Standard_tagged_immediate v
            (arithmetic_steps (int_width Standard_tagged_immediate) offset
               slope))).
  - exact
      (bind_arithmetic_steps_binds_again Standard_naked_immediate v
         (arithmetic_steps (int_width Standard_naked_immediate) offset
            slope)).
Qed.

Lemma merged_scoped (v : variable) (i i' : Z) (ac ac' : apply_cont)
    (others : list (Z * apply_cont)) (wrap : expr -> expr) :
  binds_again v wrap ->
  scoped (Switch (Var v) ((i, ac) :: (i', ac') :: others))
    (wrap
       (Apply_cont
          (Apply_cont_expr (ac_continuation ac)
             (merged_arguments v (ac_args ac) (ac_args ac')) None))).
Proof.
  intros Hwrap _.
  destruct (Hwrap
    (Apply_cont
       (Apply_cont_expr (ac_continuation ac)
          (merged_arguments v (ac_args ac) (ac_args ac')) None)))
    as (Hclosed & Hv & Hk).
  rewrite Hclosed, Hk. split; [ reflexivity | split ].
  - assert
      (Hswitch :
        ∀ y : variable,
          y = v ∨ In y (free_variables_of_simples (ac_args ac)) ->
          In y
            (free_variables_of_expr
               (Switch (Var v) ((i, ac) :: (i', ac') :: others)))).
    { intros y [-> | Hy]; scope; simpl;
        [ left; reflexivity | right; apply in_or_app; left; exact Hy ]. }
    intros x Hx. apply Hswitch. apply Hv in Hx.
    destruct Hx as [<- | Hx]; [ left; reflexivity | ].
    scope_in Hx. apply merged_arguments_free in Hx.
    destruct Hx as [<- | Hx]; [ left; reflexivity | right; exact Hx ].
  - scope. simpl. intros k [<- | []]. left. reflexivity.
Qed.

Lemma merge_switch_scoped (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  scoped (Switch scrutinee arms) (merge_switch scrutinee arms).
Proof.
  unfold merge_switch.
  destruct scrutinee as [v | sym | c]; try apply scoped_refl.
  destruct arms as [ | [i ac] [ | [i' ac'] others]]; try apply scoped_refl.
  cbv zeta.
  destruct (var_in_simples v _); [ | apply scoped_refl ].
  destruct (forallb (arm_is_merged By_tagging v _) _).
  - apply (merged_scoped v i i' ac ac' others (bind_tagged v)).
    apply bind_tagged_binds_again.
  - destruct (forallb (arm_is_merged By_tagging_and_negating v _) _).
    + apply
        (merged_scoped v i i' ac ac' others
           (fun body => bind_tagged v (bind_negated v body))).
      apply binds_again_twice;
        [ apply bind_tagged_binds_again | apply bind_negated_binds_again ].
    + destruct (forallb _ _).
      * apply (merged_scoped v i i' ac ac' others (fun body => body)).
        apply binds_again_nothing.
      * destruct (arithmetic_of_arms _ _ _ _) as [[[kind offset] slope] | ];
          [ | apply scoped_refl ].
        destruct (_ && _); [ | apply scoped_refl ].
        apply
          (merged_scoped v i i' ac ac' others
             (bind_arithmetic kind offset slope v)).
        apply bind_arithmetic_binds_again.
Qed.

(* The ways chosen for each argument. A variable is bound to something
   computed from the scrutinee, which is not bound again; the
   variables are those that the rewrite is given, and what is under
   their [Let]s mentions them where an argument is computed. *)
Definition binds_from (v x : variable) (wrap : expr -> expr) : Prop :=
  ∀ body : expr,
    code_closed (wrap body) = code_closed body
    ∧ (∀ y : variable,
         In y (free_variables_of_expr (wrap body)) ->
         y = v ∨ (In y (free_variables_of_expr body) ∧ y <> x))
    ∧ free_continuations_of_expr (wrap body)
      = free_continuations_of_expr body.

Lemma bind_tagged_to_binds_from (v x : variable) :
  binds_from v x (bind_tagged_to v x).
Proof.
  intros body. split; [ reflexivity | split; [ | reflexivity ] ].
  unfold bind_tagged_to. scope. simpl.
  intros y [<- | Hy]; [ left; reflexivity | right ].
  apply in_without in Hy. destruct Hy as [Hy Hne].
  split; [ exact Hy | ]. intros ->. apply Hne. left. reflexivity.
Qed.

Lemma bind_same_to_binds_from (v x : variable) :
  binds_from v x (bind_same_to v x).
Proof.
  intros body. split; [ reflexivity | split; [ | reflexivity ] ].
  unfold bind_same_to. scope. simpl.
  intros y [<- | Hy]; [ left; reflexivity | right ].
  apply in_without in Hy. destruct Hy as [Hy Hne].
  split; [ exact Hy | ]. intros ->. apply Hne. left. reflexivity.
Qed.

Lemma binds_from_then (v x : variable) (wrap wrap' : expr -> expr) :
  binds_from v x wrap -> binds_again x wrap' ->
  binds_from v x (fun body => wrap (wrap' body)).
Proof.
  intros Hwrap Hwrap' body.
  destruct (Hwrap (wrap' body)) as (Hclosed & Hv & Hk).
  destruct (Hwrap' body) as (Hclosed' & Hv' & Hk').
  split; [ congruence | split; [ | congruence ] ].
  intros y Hy. apply Hv in Hy.
  destruct Hy as [-> | [Hy Hne]]; [ left; reflexivity | right ].
  split; [ | exact Hne ].
  apply Hv' in Hy. destruct Hy as [Heq | Hy]; [ | exact Hy ].
  elim Hne. symmetry. exact Heq.
Qed.

Lemma bind_way_binds_from (how : merge_by) (v x : variable) :
  how <> As_it_is -> binds_from v x (bind_way how v x).
Proof.
  destruct how as [ | | | [ | ] offset slope]; intros Hhow.
  - apply bind_tagged_to_binds_from.
  - exact
      (binds_from_then v x _ _ (bind_tagged_to_binds_from v x)
         (bind_negated_binds_again x)).
  - elim Hhow. reflexivity.
  - exact
      (binds_from_then v x _ _ (bind_tagged_to_binds_from v x)
         (bind_arithmetic_steps_binds_again Standard_tagged_immediate x
            (arithmetic_steps (int_width Standard_tagged_immediate) offset
               slope))).
  - exact
      (binds_from_then v x _ _ (bind_same_to_binds_from v x)
         (bind_arithmetic_steps_binds_again Standard_naked_immediate x
            (arithmetic_steps (int_width Standard_naked_immediate) offset
               slope))).
Qed.

(* The names that the [Let]s of the ways bind. *)
Fixpoint names_bound (names : list variable) (ways : list argument_way)
    : list variable :=
  match names, ways with
  | x :: names, way :: ways =>
    match way with
    | Same_in_every_arm _ | Computed As_it_is => names_bound names ways
    | Computed (By_tagging | By_tagging_and_negating | By_arithmetic _ _ _) =>
      x :: names_bound names ways
    end
  | _, _ => []
  end.

Lemma bind_ways_scope (v : variable) (body : expr) (names : list variable) :
  ∀ ways : list argument_way,
    code_closed (bind_ways v names ways body) = code_closed body
    ∧ (∀ y : variable,
         In y (free_variables_of_expr (bind_ways v names ways body)) ->
         y = v
         ∨ (In y (free_variables_of_expr body)
            ∧ ~ In y (names_bound names ways)))
    ∧ free_continuations_of_expr (bind_ways v names ways body)
      = free_continuations_of_expr body.
Proof.
  induction names as [ | x names IH]; intros [ | way ways];
    cbn [bind_ways names_bound];
    try
      (split; [ reflexivity | split; [ | reflexivity ] ];
       intros y Hy; right; split; [ exact Hy | intros [] ]).
  destruct (IH ways) as (Hclosed & Hv & Hk).
  assert
    (Hbinds :
      ∀ how : merge_by,
        how <> As_it_is ->
        code_closed (bind_way how v x (bind_ways v names ways body))
        = code_closed body
        ∧ (∀ y : variable,
             In y
               (free_variables_of_expr
                  (bind_way how v x (bind_ways v names ways body))) ->
             y = v
             ∨ (In y (free_variables_of_expr body)
                ∧ ~ In y (x :: names_bound names ways)))
        ∧ free_continuations_of_expr
            (bind_way how v x (bind_ways v names ways body))
          = free_continuations_of_expr body).
  { intros how Hhow.
    destruct (bind_way_binds_from how v x Hhow (bind_ways v names ways body))
      as (Hclosed' & Hv' & Hk').
    split; [ congruence | split; [ | congruence ] ].
    intros y Hy. apply Hv' in Hy.
    destruct Hy as [-> | [Hy Hne]]; [ left; reflexivity | ].
    apply Hv in Hy.
    destruct Hy as [-> | [Hy Hbound]]; [ left; reflexivity | right ].
    split; [ exact Hy | ].
    intros [Heq | Hin]; [ apply Hne; symmetry; exact Heq | exact (Hbound Hin) ].
  }
  destruct way as [s | [ | | | kind offset slope]];
    first
      [ apply Hbinds; discriminate
      | split; [ exact Hclosed | split; [ exact Hv | exact Hk ] ] ].
Qed.

(* The merged arguments are the scrutinee, the names that are bound,
   and what the arguments of an arm mention. *)
Lemma merged_by_ways_free (v : variable) (i : Z) (names : list variable) :
  ∀ (ways : list argument_way) (args : list simple),
    arguments_in_arm i ways = Some args ->
    ∀ y : variable,
      In y (free_variables_of_simples (merged_by_ways v names ways)) ->
      y = v
      ∨ In y (names_bound names ways)
      ∨ In y (free_variables_of_simples args).
Proof.
  induction names as [ | x names IH]; intros [ | way ways] args Hargs y Hy;
    cbn [merged_by_ways] in Hy; try (destruct Hy).
  cbn [arguments_in_arm] in Hargs.
  destruct (argument_in_arm i way) as [s0 | ] eqn:Hs0; [ | discriminate ].
  destruct (arguments_in_arm i ways) as [args0 | ] eqn:Hargs0;
    [ | discriminate ].
  injection Hargs as <-.
  unfold free_variables_of_simples in Hy. cbn [flat_map] in Hy.
  apply in_app_or in Hy.
  unfold free_variables_of_simples. cbn [flat_map names_bound].
  destruct Hy as [Hy | Hy].
  - destruct way as [s | how]; cbn [argument_in_arm] in Hs0.
    + injection Hs0 as <-. right. right. apply in_or_app. left. exact Hy.
    + destruct how; simpl in Hy; destruct Hy as [<- | []];
        first [ left; reflexivity | right; left; left; reflexivity ].
  - destruct (IH ways args0 Hargs0 y Hy) as [-> | [Hin | Hin]].
    + left. reflexivity.
    + right. left.
      destruct way as [s | [ | | | kind offset slope]];
        first [ exact Hin | right; exact Hin ].
    + right. right. apply in_or_app. right. exact Hin.
Qed.

Lemma merge_switch_by_arguments_scoped (names : list variable)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (merged : expr) :
  merge_switch_by_arguments names scrutinee arms = Some merged ->
  scoped (Switch scrutinee arms) merged.
Proof.
  unfold merge_switch_by_arguments.
  destruct scrutinee as [v | sym | c]; try discriminate.
  destruct arms as [ | [i0 [k0 args trap]] [ | arm1 others]];
    try discriminate.
  cbv zeta. cbn [ac_continuation ac_args].
  destruct (ways_of_arguments _ _) as [ways | ]; [ | discriminate ].
  destruct (names_usable v names ways && _) eqn:Hok; [ | discriminate ].
  intros [= <-] _.
  apply andb_prop in Hok as [_ Hall].
  cbn [forallb] in Hall. apply andb_prop in Hall as [Hfirst _].
  unfold arm_has_ways in Hfirst. cbn [fst snd] in Hfirst.
  destruct (arguments_in_arm i0 ways) as [args0 | ] eqn:Hargs0;
    [ | discriminate ].
  destruct (apply_cont_eq_dec _ _) as [Heq | ]; [ | discriminate ].
  injection Heq as -> ->.
  destruct
    (bind_ways_scope v
       (Apply_cont (Apply_cont_expr k0 (merged_by_ways v names ways) None))
       names ways)
    as (Hclosed & Hv & Hk).
  rewrite Hclosed, Hk. split; [ reflexivity | split ].
  - assert
      (Hswitch :
        ∀ y : variable,
          y = v ∨ In y (free_variables_of_simples args0) ->
          In y
            (free_variables_of_expr
               (Switch (Var v)
                  ((i0, Apply_cont_expr k0 args0 None) :: arm1 :: others)))).
    { intros y [-> | Hy]; scope; simpl;
        [ left; reflexivity | right; apply in_or_app; left; exact Hy ]. }
    intros y Hy. apply Hswitch. apply Hv in Hy.
    destruct Hy as [-> | [Hy Hbound]]; [ left; reflexivity | ].
    scope_in Hy.
    destruct (merged_by_ways_free v i0 names ways args0 Hargs0 y Hy)
      as [-> | [Hin | Hin]];
      [ left; reflexivity | contradiction | right; exact Hin ].
  - scope. simpl. intros k [<- | []]. left. reflexivity.
Qed.

Lemma merge_switch_knowing_scoped (known : known_continuations)
    (scrutinee : simple) (arms : list (Z * apply_cont)) :
  scoped (Switch scrutinee arms) (merge_switch_knowing known scrutinee arms).
Proof.
  unfold merge_switch_knowing.
  pose proof (merge_switch_scoped scrutinee arms) as Hone.
  destruct (merge_switch scrutinee arms); try exact Hone.
  destruct arms as [ | [i ac] others]; [ apply scoped_refl | ].
  destruct (lookup (ac_continuation ac) known) as [names | ];
    [ | apply scoped_refl ].
  destruct (merge_switch_by_arguments names scrutinee ((i, ac) :: others))
    as [merged | ] eqn:Hmerged;
    [ exact (merge_switch_by_arguments_scoped _ _ _ _ Hmerged)
    | apply scoped_refl ].
Qed.

Fixpoint merge_switches_knowing_scoped (known : known_continuations)
    (e : expr) {struct e} :
  scoped e (merge_switches_knowing known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [merge_switches_knowing];
    first
      [ apply merge_switch_knowing_scoped
      | scoped_under_binders merge_switches_knowing_scoped ].
Qed.

Lemma merge_switches_scoped (e : expr) : scoped e (merge_switches e).
Proof. exact (merge_switches_knowing_scoped [] e). Qed.

Theorem merge_switches_keeps_scope : merge_switches_keeps_scope_statement.
Proof. apply keeps_scope_intro, merge_switches_scoped. Qed.

(* Continuations that are not used. *)

Fixpoint remove_unused_conts_scoped (e : expr) :
  scoped e (remove_unused_conts e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [remove_unused_conts remove_unused_conts_in_handler];
    try (scoped_under_binders remove_unused_conts_scoped).
  destruct (cont_free_b k (remove_unused_conts body)) eqn:Hfree.
  - scoped_under_binders remove_unused_conts_scoped.
  - apply scoped_let_cont_unused;
      [ apply remove_unused_conts_scoped | exact Hfree ].
Qed.

Theorem remove_unused_conts_keeps_scope :
  remove_unused_conts_keeps_scope_statement.
Proof. apply keeps_scope_intro, remove_unused_conts_scoped. Qed.

(* [Let]s that are not used. *)

Fixpoint remove_unused_lets_scoped (e : expr) :
  scoped e (remove_unused_lets e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [remove_unused_lets remove_unused_lets_in_handler];
    try (scoped_under_binders remove_unused_lets_scoped).
  destruct (var_free_b x (remove_unused_lets body)) eqn:Hfree; cbn [orb];
    [ scoped_under_binders remove_unused_lets_scoped | ].
  destruct (negb (named_may_go d));
    [ scoped_under_binders remove_unused_lets_scoped | ].
  apply scoped_let_unused; [ apply remove_unused_lets_scoped | exact Hfree ].
Qed.

Theorem remove_unused_lets_keeps_scope :
  remove_unused_lets_keeps_scope_statement.
Proof. apply keeps_scope_intro, remove_unused_lets_scoped. Qed.

Fixpoint remove_unused_defined_lets_scoped (e : expr) :
  ∀ kinds : kinds_in_scope, scoped e (remove_unused_defined_lets kinds e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros kinds; cbn [remove_unused_defined_lets];
    try (scoped_under_binders remove_unused_defined_lets_scoped).
  destruct (named_defined kinds kind d); cbn [andb];
    [ | scoped_under_binders remove_unused_defined_lets_scoped ].
  destruct
    (var_free_b x (remove_unused_defined_lets ((x, kind) :: kinds) body))
    eqn:Hfree; cbn [negb];
    [ scoped_under_binders remove_unused_defined_lets_scoped | ].
  apply scoped_let_unused;
    [ apply remove_unused_defined_lets_scoped | exact Hfree ].
Qed.

Theorem remove_unused_defined_lets_keeps_scope :
  remove_unused_defined_lets_keeps_scope_statement.
Proof.
  intros kinds. apply keeps_scope_intro. intros e.
  apply remove_unused_defined_lets_scoped.
Qed.

(* The identities of arithmetic. *)

Lemma named_of_outcome_free (kind : standard_int) (o : outcome)
    (other : simple) (d : named) :
  named_of_outcome kind o other = Some d ->
  incl (free_variables_of_named d) (free_variables_of_simple other).
Proof.
  destruct o; cbn [named_of_outcome]; intros Hd; try discriminate;
    injection Hd as <-; cbn [free_variables_of_named];
    unfold free_variables_of_simples; cbn [flat_map free_variables_of_simple];
    rewrite ?app_nil_r; first [ apply incl_refl | intros a [] ].
Qed.

Lemma identity_of_binary_free (p : binary_primitive) (arg1 arg2 : simple)
    (d : named) :
  identity_of_binary p arg1 arg2 = Some d ->
  incl (free_variables_of_named d)
    (free_variables_of_simple arg1 ++ free_variables_of_simple arg2).
Proof.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [identity_of_binary]; try discriminate.
  - destruct arg1 as [x | | ], arg2 as [y | | ]; try discriminate.
    destruct (Pos.eqb x y); [ | discriminate ].
    intros Hd. injection Hd as <-. intros a [].
  - destruct (int_of_simple kind arg1), (int_of_simple kind arg2);
      try discriminate.
    + destruct (is_const arg2); [ discriminate | ].
      intros Hd. apply incl_appr. exact (named_of_outcome_free _ _ _ _ Hd).
    + destruct (is_const arg1); [ discriminate | ].
      intros Hd. apply incl_appl. exact (named_of_outcome_free _ _ _ _ Hd).
  - destruct (int_of_simple kind arg1),
      (int_of_simple Standard_naked_immediate arg2);
      try discriminate.
    + destruct (is_const arg2); [ discriminate | ].
      intros Hd. apply incl_appr. exact (named_of_outcome_free _ _ _ _ Hd).
    + destruct (is_const arg1); [ discriminate | ].
      intros Hd. apply incl_appl. exact (named_of_outcome_free _ _ _ _ Hd).
Qed.

Lemma identity_named_free (d : named) :
  incl (free_variables_of_named (identity_named d))
    (free_variables_of_named d).
Proof.
  destruct d as [s | p args | | | | | ]; try apply incl_refl.
  unfold identity_named.
  destruct (binary_applied p args) as [[[p' arg1] arg2] | ] eqn:Happlied;
    [ | apply incl_refl ].
  destruct (identity_of_binary p' arg1 arg2) as [d' | ] eqn:Hidentity;
    [ | apply incl_refl ].
  unfold binary_applied in Happlied.
  destruct p as [ | p0 | | | ]; try discriminate.
  destruct args as [ | a1 [ | a2 [ | a3 args]]]; try discriminate.
  injection Happlied as -> -> ->.
  cbn [free_variables_of_named]. unfold free_variables_of_simples.
  cbn [flat_map]. rewrite app_nil_r.
  exact (identity_of_binary_free _ _ _ _ Hidentity).
Qed.

Fixpoint apply_identities_scoped (e : expr) : scoped e (apply_identities e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [apply_identities apply_identities_in_handler];
    first
      [ apply scoped_let;
        [ apply identity_named_free | apply apply_identities_scoped ]
      | scoped_under_binders apply_identities_scoped ].
Qed.

Theorem apply_identities_keeps_scope : apply_identities_keeps_scope_statement.
Proof. apply keeps_scope_intro, apply_identities_scoped. Qed.

(* A continuation applied once, at the end of its scope. *)

Lemma inline_cont_at_tail_scoped (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (body : expr) :
  ∀ inlined : expr,
    inline_cont_at_tail k params metadata handler body = Some inlined ->
    code_closed handler = true ->
    code_closed body = true ->
    code_closed inlined = true
    ∧ incl (free_variables_of_expr inlined)
        (without params (free_variables_of_expr handler)
         ++ free_variables_of_expr body)
    ∧ incl (free_continuations_of_expr inlined)
        (free_continuations_of_expr handler).
Proof.
  induction body
    as [x kind d body IH | bindings body IH | | | | ac | | ];
    intros inlined Hinlined Hhandler Hbody;
    cbn [inline_cont_at_tail] in Hinlined; try discriminate; scope_in Hbody.
  - destruct (var_free_in_handler_b x (Handler params metadata handler));
      [ discriminate | ].
    destruct (inline_cont_at_tail k params metadata handler body)
      as [inlined' | ]; [ | discriminate ].
    injection Hinlined as <-.
    destruct (IH inlined' eq_refl Hhandler Hbody) as (Hclosed & Hv & Hk).
    scope. split; [ exact Hclosed | split; [ | exact Hk ] ].
    intros y Hy. apply in_app_or in Hy.
    destruct Hy as [Hy | Hy];
      [ apply in_or_app; right; apply in_or_app; left; exact Hy | ].
    apply in_without in Hy. destruct Hy as [Hy Hx].
    apply Hv, in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
    apply in_or_app. right. apply in_without. split; assumption.
  - destruct (inline_cont_at_tail k params metadata handler body)
      as [inlined' | ]; [ | discriminate ].
    injection Hinlined as <-.
    apply andb_true_iff in Hbody. destruct Hbody as [Hgroup Hbody].
    destruct (IH inlined' eq_refl Hhandler Hbody) as (Hclosed & Hv & Hk).
    scope. rewrite Hgroup, Hclosed.
    split; [ reflexivity | split ].
    + intros y Hy. apply in_app_or in Hy. apply in_or_app.
      destruct Hy as [Hy | Hy];
        [ right; apply in_or_app; left; exact Hy | ].
      apply Hv, in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; exact Hy | right; apply in_or_app; right; exact Hy ].
    + rewrite (bindings_closed_no_continuations bindings Hgroup). exact Hk.
  - destruct
      (Pos.eqb (ac_continuation ac) k && no_trap_action ac
       && Nat.eqb (length params) (length (ac_args ac))
       && forallb (fun param => negb (var_in_simples param (ac_args ac)))
            params)
      eqn:Htest; [ | discriminate ].
    injection Hinlined as <-.
    apply andb_true_iff in Htest. destruct Htest as [Htest _].
    apply andb_true_iff in Htest. destruct Htest as [_ Hlength].
    apply Nat.eqb_eq in Hlength.
    rewrite let_params_code_closed, let_params_free_continuations.
    split; [ exact Hhandler | split; [ | apply incl_refl ] ].
    intros y Hy. apply let_params_free_variables in Hy; [ | exact Hlength ].
    apply in_or_app.
    destruct Hy as [Hy | [Hy Hparams]];
      [ right; exact Hy | left; apply in_without; split; assumption ].
Qed.

Fixpoint inline_tail_conts_scoped (e : expr) : scoped e (inline_tail_conts e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [inline_tail_conts];
    try (scoped_under_binders inline_tail_conts_scoped).
  destruct
    (inline_cont_at_tail k params metadata (inline_tail_conts handler)
       (inline_tail_conts body))
    as [inlined | ] eqn:Hinlined;
    [ | scoped_under_binders inline_tail_conts_scoped ].
  intros Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hhandler Hbody].
  destruct (inline_tail_conts_scoped handler Hhandler)
    as (Hhandler' & Hv_handler & Hk_handler).
  destruct (inline_tail_conts_scoped body Hbody)
    as (Hbody' & Hv_body & Hk_body).
  destruct
    (inline_cont_at_tail_scoped k params metadata _ _ inlined Hinlined
       Hhandler' Hbody')
    as (Hclosed & Hv & Hk).
  split; [ exact Hclosed | split ].
  - eapply incl_tran; [ exact Hv | ].
    apply incl_app_app; [ apply incl_without, Hv_handler | exact Hv_body ].
  - apply incl_appl. eapply incl_tran; [ exact Hk | exact Hk_handler ].
Qed.

Theorem inline_tail_conts_keeps_scope : inline_tail_conts_keeps_scope_statement.
Proof. apply keeps_scope_intro, inline_tail_conts_scoped. Qed.

(* A switch whose arm goes to a switch on the same scrutinee. *)

(* An arm of the body as it is, or one of the handler. *)
Lemma thread_arm_from (k : continuation) (inner : list (Z * apply_cont))
    (arm : Z * apply_cont) :
  snd (thread_arm k inner arm) = snd arm
  ∨ ∃ j : Z, In (j, snd (thread_arm k inner arm)) inner.
Proof.
  destruct arm as [i ac]. unfold thread_arm.
  destruct (ac_args ac); [ | left; reflexivity ].
  destruct (Pos.eqb (ac_continuation ac) k && no_trap_action ac);
    [ | left; reflexivity ].
  destruct (find_arm i inner) as [ac' | ] eqn:Hfind; [ | left; reflexivity ].
  destruct (among k (apply_cont_continuations ac'));
    [ left; reflexivity | ].
  right. exact (find_arm_In i inner ac' Hfind).
Qed.

Lemma thread_arms_names (f : apply_cont -> list positive) (k : continuation)
    (inner arms : list (Z * apply_cont)) :
  incl (flat_map (fun arm => f (snd arm)) (map (thread_arm k inner) arms))
    (flat_map (fun arm => f (snd arm)) inner
     ++ flat_map (fun arm => f (snd arm)) arms).
Proof.
  intros a Ha. apply in_flat_map in Ha. destruct Ha as (arm' & Harm' & Ha).
  apply in_map_iff in Harm'. destruct Harm' as (arm & <- & Harm).
  apply in_or_app.
  destruct (thread_arm_from k inner arm) as [Hsame | [j Hinner]].
  - right. apply in_flat_map. exists arm. rewrite <- Hsame.
    split; [ exact Harm | exact Ha ].
  - left. apply in_flat_map. exists (j, snd (thread_arm k inner arm)).
    split; [ exact Hinner | exact Ha ].
Qed.

Lemma arms_apply_none (k : continuation) (arms : list (Z * apply_cont)) :
  arms_apply k arms = false ->
  ~ In k (flat_map (fun arm => apply_cont_continuations (snd arm)) arms).
Proof.
  intros Happly Hin. apply in_flat_map in Hin.
  destruct Hin as (arm & Harm & Hk).
  assert (Hsome : arms_apply k arms = true).
  { apply existsb_exists. exists arm.
    split; [ exact Harm | apply among_In; exact Hk ]. }
  congruence.
Qed.

Lemma thread_switch_scoped (k : continuation) (h : continuation_handler)
    (body : expr) :
  scoped (Let_cont_non_recursive k h body) (thread_switch k h body).
Proof.
  unfold thread_switch.
  destruct h as [[ | param params] metadata handler]; destruct handler;
    destruct body; try apply scoped_refl.
  destruct (hm_params_kinds metadata); [ | apply scoped_refl ].
  match goal with
  | |- context [simple_eq_dec ?s ?s'] =>
    destruct (simple_eq_dec s s') as [<- | ]; [ | apply scoped_refl ]
  end.
  cbv zeta. intros _.
  match goal with
  | |- context [map (thread_arm k ?inner) ?arms] =>
    pose proof (thread_arms_names free_variables_of_apply_cont k inner arms)
      as Hv;
    pose proof (thread_arms_names apply_cont_continuations k inner arms)
      as Hk;
    destruct (arms_apply k (map (thread_arm k inner) arms)) eqn:Happly
  end;
    scope; (split; [ reflexivity | split ]).
  - intros x Hx. apply in_app_or in Hx. apply in_or_app.
    destruct Hx as [Hx | Hx]; [ left; exact Hx | ].
    apply in_app_or in Hx.
    destruct Hx as [Hx | Hx]; [ right; apply in_or_app; left; exact Hx | ].
    apply Hv, in_app_or in Hx.
    destruct Hx as [Hx | Hx];
      [ left | right; apply in_or_app; right; exact Hx ].
    apply in_without. split; [ | intros [] ]. apply in_or_app. right. exact Hx.
  - intros j Hj. apply in_app_or in Hj. apply in_or_app.
    destruct Hj as [Hj | Hj]; [ left; exact Hj | ].
    apply in_without in Hj. destruct Hj as [Hj Hout].
    apply Hk, in_app_or in Hj.
    destruct Hj as [Hj | Hj];
      [ left; exact Hj | right; apply in_without; split; assumption ].
  - intros x Hx. apply in_app_or in Hx. apply in_or_app.
    destruct Hx as [Hx | Hx]; [ right; apply in_or_app; left; exact Hx | ].
    apply Hv, in_app_or in Hx.
    destruct Hx as [Hx | Hx];
      [ left | right; apply in_or_app; right; exact Hx ].
    apply in_without. split; [ | intros [] ]. apply in_or_app. right. exact Hx.
  - intros j Hj. apply in_or_app.
    pose proof (arms_apply_none k _ Happly) as Hnone.
    assert (Hout : ~ In j [k])
      by (intros [<- | []]; exact (Hnone Hj)).
    apply Hk, in_app_or in Hj.
    destruct Hj as [Hj | Hj];
      [ left; exact Hj | right; apply in_without; split; assumption ].
Qed.

Fixpoint thread_switches_scoped (e : expr) : scoped e (thread_switches e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [thread_switches];
    try (scoped_under_binders thread_switches_scoped).
  eapply scoped_trans; [ | apply thread_switch_scoped ].
  scoped_under_binders thread_switches_scoped.
Qed.

Theorem thread_switches_keeps_scope : thread_switches_keeps_scope_statement.
Proof. apply keeps_scope_intro, thread_switches_scoped. Qed.

(* Groups of static bindings. *)

Fixpoint drop_deleted_code_scoped (e : expr) : scoped e (drop_deleted_code e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [drop_deleted_code];
    first
      [ apply scoped_let_static_if_any;
        [ apply scoped_bindings_filter | apply drop_deleted_code_scoped ]
      | scoped_under_binders drop_deleted_code_scoped ].
Qed.

Theorem drop_deleted_code_keeps_scope : drop_deleted_code_keeps_scope_statement.
Proof. apply keeps_scope_intro, drop_deleted_code_scoped. Qed.

Fixpoint merge_statics_scoped (e : expr) : scoped e (merge_statics e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [merge_statics]; try (scoped_under_binders merge_statics_scoped).
  pose proof (merge_statics_scoped body) as Hbody.
  destruct (merge_statics body)
    as [ | bindings' body' | | | | | | ];
    try (apply scoped_let_static;
         [ apply scoped_bindings_code_first | exact Hbody ]).
  eapply scoped_trans;
    [ apply scoped_let_static; [ apply scoped_bindings_refl | exact Hbody ] | ].
  eapply scoped_trans; [ apply scoped_let_static_merge | ].
  apply scoped_let_static;
    [ apply scoped_bindings_code_first | apply scoped_refl ].
Qed.

Theorem merge_statics_keeps_scope : merge_statics_keeps_scope_statement.
Proof. apply keeps_scope_intro, merge_statics_scoped. Qed.

(* A rewrite taken into the bodies of code. *)

Section Rewrite_code.

Variable f : expr -> expr.

Hypothesis f_keeps_scope : keeps_scope f.

Lemma rewrite_code_body_scoped (body : expr) :
  scoped body (rewrite_code_body f body).
Proof.
  unfold rewrite_code_body. cbv zeta.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f body)));
    [ apply keeps_scope_scoped, f_keeps_scope | apply scoped_refl ].
Qed.

Lemma rewrite_code_in_bindings_scoped (bindings : list static_binding) :
  scoped_bindings bindings (map (rewrite_code_in_binding f) bindings).
Proof.
  induction bindings as [ | binding bindings IH]; cbn [map];
    [ apply scoped_bindings_nil | ].
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    cbn [rewrite_code_in_binding];
    [ apply scoped_bindings_cons_code;
      [ apply scoped_code_intro, rewrite_code_body_scoped | exact IH ]
    | apply scoped_bindings_cons_same, IH
    | apply scoped_bindings_cons_same, IH ].
Qed.

Fixpoint rewrite_code_scoped (e : expr) : scoped e (rewrite_code f e)
with rewrite_code_in_handler_scoped (h : continuation_handler) :
  scoped_handler h (rewrite_code_in_handler f h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rewrite_code rewrite_code_in_handler].
    + apply scoped_let; [ apply incl_refl | apply rewrite_code_scoped ].
    + apply scoped_let_static;
        [ apply rewrite_code_in_bindings_scoped | apply rewrite_code_scoped ].
    + apply scoped_let_cont;
        [ apply rewrite_code_in_handler_scoped | apply rewrite_code_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply rewrite_code_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply rewrite_code_in_handler_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler].
    cbn [rewrite_code rewrite_code_in_handler].
    apply scoped_handler_intro, rewrite_code_scoped.
Qed.

End Rewrite_code.

Theorem rewrite_code_keeps_scope : rewrite_code_keeps_scope_statement.
Proof.
  intros f Hf. apply keeps_scope_intro. intros e.
  exact (rewrite_code_scoped f Hf e).
Qed.

(* ======================================================================== *)
(* The rewrites that go by what is known                                    *)
(* ======================================================================== *)

Lemma unary_applied_some (p : prim) (args : list simple)
    (p' : unary_primitive) (arg : simple) :
  unary_applied p args = Some (p', arg) -> p = Unary p' ∧ args = [arg].
Proof.
  unfold unary_applied. destruct p; try discriminate.
  destruct args as [ | a [ | b args]]; try discriminate.
  intros Happlied. injection Happlied as -> ->. split; reflexivity.
Qed.

(* Loading from a known block. *)

Definition block_within (names : list variable) (block : known_block)
    : Prop :=
  ∀ s : simple,
    In (Some s) (snd block) -> incl (free_variables_of_simple s) names.

Definition blocks_within (names : list variable) (known : known_blocks)
    : Prop :=
  (∀ entry : variable * known_block,
    In entry (known_vars known) -> block_within names (snd entry))
  ∧ (∀ entry : symbol * known_block,
      In entry (known_symbols known) -> block_within names (snd entry)).

Lemma blocks_within_weaken (names names' : list variable)
    (known : known_blocks) :
  incl names names' -> blocks_within names known -> blocks_within names' known.
Proof.
  intros Hincl [Hvars Hsymbols].
  split; intros entry Hentry s Hs;
    (eapply incl_tran; [ | exact Hincl ]);
    [ exact (Hvars entry Hentry s Hs) | exact (Hsymbols entry Hentry s Hs) ].
Qed.

Lemma forget_in_block_within (names : list variable) (x : variable)
    (block : known_block) :
  block_within names block -> block_within names (forget_in_block x block).
Proof.
  intros Hblock s Hs. unfold forget_in_block in Hs. cbn [snd] in Hs.
  apply in_map_iff in Hs. destruct Hs as (field & Hfield & Hin).
  destruct field as [s' | ]; cbn [forget_field] in Hfield; [ | discriminate ].
  destruct (var_in_simple x s'); [ discriminate | ].
  injection Hfield as ->. exact (Hblock s Hin).
Qed.

Lemma forget_var_within (names : list variable) (x : variable)
    (known : known_blocks) :
  blocks_within names known -> blocks_within names (forget_var x known).
Proof.
  intros [Hvars Hsymbols]. unfold forget_var.
  split; cbn [known_vars known_symbols]; intros entry Hentry;
    apply in_map_iff in Hentry; destruct Hentry as (entry' & <- & Hin);
    cbn [snd]; apply forget_in_block_within.
  - apply filter_In in Hin. exact (Hvars entry' (proj1 Hin)).
  - exact (Hsymbols entry' Hin).
Qed.

Lemma forget_vars_within (names : list variable) (xs : list variable) :
  ∀ known : known_blocks,
    blocks_within names known -> blocks_within names (forget_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known Hknown; cbn [forget_vars];
    [ exact Hknown | apply IH, forget_var_within, Hknown ].
Qed.

Lemma known_under_let_within (names : list variable) (known : known_blocks)
    (x : variable) (d : named) :
  blocks_within names known ->
  blocks_within (x :: names ++ free_variables_of_named d)
    (known_under_let known x d).
Proof.
  intros Hknown. unfold known_under_let. cbv zeta.
  assert (Hforgot :
    blocks_within (x :: names ++ free_variables_of_named d)
      (forget_var x known)).
  { apply forget_var_within.
    eapply blocks_within_weaken; [ | exact Hknown ].
    apply incl_tl, incl_appl, incl_refl. }
  destruct (block_made d) as [block | ] eqn:Hmade; [ | exact Hforgot ].
  destruct Hforgot as [Hvars Hsymbols].
  split; cbn [known_vars known_symbols]; [ | exact Hsymbols ].
  intros entry [<- | Hentry]; [ | exact (Hvars entry Hentry) ].
  cbn [snd]. apply forget_in_block_within.
  unfold block_made in Hmade.
  repeat
    match type of Hmade with
    | match ?t with _ => _ end = Some _ => destruct t; try discriminate Hmade
    end.
  injection Hmade as <-. intros s Hs. cbn [snd] in Hs.
  apply in_map_iff in Hs. destruct Hs as (s' & Heq & Hin).
  injection Heq as ->. intros y Hy. right. apply in_or_app. right.
  cbn [free_variables_of_named]. unfold free_variables_of_simples.
  apply in_flat_map. exists s. split; assumption.
Qed.

Lemma known_under_statics_within (names : list variable)
    (known : known_blocks) (bindings : list static_binding) :
  blocks_within names known ->
  blocks_within (names ++ flat_map free_variables_of_binding bindings)
    (known_under_statics known bindings).
Proof.
  intros Hknown.
  destruct
    (blocks_within_weaken names
       (names ++ flat_map free_variables_of_binding bindings) known
       (incl_appl _ (incl_refl _)) Hknown)
    as [Hvars Hsymbols].
  unfold known_under_statics.
  split; cbn [known_vars known_symbols]; [ exact Hvars | ].
  intros entry Hentry. apply in_app_or in Hentry.
  destruct Hentry as [Hentry | Hentry]; [ | exact (Hsymbols entry Hentry) ].
  apply in_flat_map in Hentry. destruct Hentry as (binding & Hbinding & Hentry).
  unfold static_block_made in Hentry.
  destruct binding as [ | | sym const]; try solve [ destruct Hentry ].
  destruct const as [ | | tag fields | | | ]; try solve [ destruct Hentry ].
  destruct Hentry as [<- | []]. cbn [snd]. intros s Hs.
  apply in_map_iff in Hs. destruct Hs as (s' & Heq & Hin).
  injection Heq as ->. intros y Hy. apply in_or_app. right.
  apply in_flat_map. exists (Bind_symbol sym (Static_const_block tag fields)).
  split; [ exact Hbinding | ].
  cbn [free_variables_of_binding free_variables_of_static_const].
  unfold free_variables_of_simples. apply in_flat_map. exists s.
  split; assumption.
Qed.

Lemma of_known_block_from (p : unary_primitive) (block : known_block)
    (result : simple) :
  of_known_block p block = Some result ->
  In (Some result) (snd block) ∨ free_variables_of_simple result = [].
Proof.
  unfold of_known_block, field_loaded.
  repeat
    match goal with
    | |- context [match ?t with _ => _ end] => destruct t eqn:?
    end;
    intros Hresult; try discriminate Hresult; injection Hresult as <-;
    first [ right; reflexivity | left; eapply nth_error_In; eassumption ].
Qed.

Lemma load_known_named_free (names : list variable) (known : known_blocks)
    (d : named) :
  blocks_within names known ->
  incl (free_variables_of_named (load_known_named known d))
    (free_variables_of_named d ++ names).
Proof.
  intros [Hvars Hsymbols]. unfold load_known_named.
  destruct d as [ | p args | | | | | ]; try (apply incl_appl, incl_refl).
  destruct (unary_applied p args) as [[p' block] | ];
    [ | apply incl_appl, incl_refl ].
  destruct (known_block_of known block) as [known_block | ] eqn:Hknown;
    [ | apply incl_appl, incl_refl ].
  destruct (of_known_block p' known_block) as [result | ] eqn:Hresult;
    [ | apply incl_appl, incl_refl ].
  cbn [free_variables_of_named]. apply incl_appr.
  assert (Hblock : block_within names known_block).
  { destruct block as [x | sym | c]; cbn [known_block_of] in Hknown;
      [ | | discriminate ].
    - exact (Hvars (x, known_block) (lookup_in _ _ _ Hknown)).
    - exact (Hsymbols (sym, known_block) (lookup_in _ _ _ Hknown)). }
  destruct (of_known_block_from _ _ _ Hresult) as [Hin | Hnone];
    [ exact (Hblock result Hin) | rewrite Hnone; intros a [] ].
Qed.

Fixpoint load_blocks_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : known_blocks),
    blocks_within names known ->
    scoped_under names e (load_blocks_knowing known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown;
    cbn [load_blocks_knowing load_blocks_in_handler];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply load_known_named_free, Hknown | ].
    apply load_blocks_knowing_scoped, known_under_let_within, Hknown.
  - apply scoped_under_let_static.
    apply load_blocks_knowing_scoped, known_under_statics_within, Hknown.
  - apply scoped_under_let_cont; apply load_blocks_knowing_scoped;
      [ apply forget_vars_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply load_blocks_knowing_scoped, Hknown.
Qed.

Lemma load_known_blocks_scoped (e : expr) : scoped e (load_known_blocks e).
Proof.
  apply scoped_under_none, load_blocks_knowing_scoped.
  split; intros entry [].
Qed.

Theorem load_known_blocks_keeps_scope : load_known_blocks_keeps_scope_statement.
Proof. apply keeps_scope_intro, load_known_blocks_scoped. Qed.

(* Unwrapping a known number. *)

Definition numbers_within (names : list variable) (known : known_numbers)
    : Prop :=
  ∀ entry : variable * (wrapping * simple),
    In entry (known_wrapped_vars known) ->
    incl (free_variables_of_simple (snd (snd entry))) names.

Lemma forget_wrapped_var_within (names names' : list variable) (x : variable)
    (known : known_numbers) :
  incl names names' ->
  numbers_within names known ->
  numbers_within names' (forget_wrapped_var x known).
Proof.
  intros Hincl Hknown entry Hentry. unfold forget_wrapped_var in Hentry.
  cbn [known_wrapped_vars] in Hentry. apply filter_In in Hentry.
  eapply incl_tran; [ exact (Hknown entry (proj1 Hentry)) | exact Hincl ].
Qed.

Lemma forget_wrapped_vars_within (names : list variable)
    (xs : list variable) :
  ∀ known : known_numbers,
    numbers_within names known ->
    numbers_within names (forget_wrapped_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known Hknown;
    cbn [forget_wrapped_vars]; [ exact Hknown | ].
  apply IH, (forget_wrapped_var_within names names), Hknown. apply incl_refl.
Qed.

Lemma number_wrapped_free (d : named) (how : wrapping) (contents : simple) :
  number_wrapped d = Some (how, contents) ->
  incl (free_variables_of_simple contents) (free_variables_of_named d).
Proof.
  unfold number_wrapped. destruct d as [ | p args | | | | | ];
    try discriminate.
  destruct (unary_applied p args) as [[p' arg] | ] eqn:Happlied;
    [ | discriminate ].
  apply unary_applied_some in Happlied. destruct Happlied as [-> ->].
  destruct p'; try discriminate; intros Hwrapped;
    injection Hwrapped as Hhow <-; cbn [free_variables_of_named];
    unfold free_variables_of_simples; cbn [flat_map]; rewrite app_nil_r;
    apply incl_refl.
Qed.

Lemma numbers_under_let_within (names : list variable)
    (known : known_numbers) (x : variable) (d : named) :
  numbers_within names known ->
  numbers_within (x :: names ++ free_variables_of_named d)
    (numbers_under_let known x d).
Proof.
  intros Hknown. unfold numbers_under_let. cbv zeta.
  assert (Hforgot :
    numbers_within (x :: names ++ free_variables_of_named d)
      (forget_wrapped_var x known))
    by (apply (forget_wrapped_var_within names); [ | exact Hknown ];
        apply incl_tl, incl_appl, incl_refl).
  destruct (number_wrapped d) as [[how contents] | ] eqn:Hwrapped;
    [ | exact Hforgot ].
  destruct (var_in_simple x contents); [ exact Hforgot | ].
  intros entry Hentry. cbn [known_wrapped_vars] in Hentry.
  destruct Hentry as [<- | Hentry]; [ | exact (Hforgot entry Hentry) ].
  cbn [snd].
  eapply incl_tran; [ exact (number_wrapped_free d how contents Hwrapped) | ].
  apply incl_tl, incl_appr, incl_refl.
Qed.

Lemma unwrapped_free (names : list variable) (known : known_numbers)
    (how : wrapping) (wrapped contents : simple) :
  numbers_within names known ->
  unwrapped known how wrapped = Some contents ->
  incl (free_variables_of_simple contents) names.
Proof.
  intros Hknown. unfold unwrapped.
  destruct wrapped as [x | sym | c]; [ | | discriminate ].
  - destruct (lookup x (known_wrapped_vars known)) as [[how' contents'] | ]
      eqn:Hlookup; [ | discriminate ].
    destruct (wrapping_eqb how how'); [ | discriminate ].
    intros Hcontents. injection Hcontents as <-.
    exact (Hknown (x, (how', contents')) (lookup_in _ _ _ Hlookup)).
  - destruct how as [kind | ]; [ | discriminate ].
    destruct (lookup sym (known_boxed_symbols known)) as [c | ];
      [ | discriminate ].
    destruct (is_number_of_kind kind c); [ | discriminate ].
    intros Hcontents. injection Hcontents as <-. intros a [].
Qed.

Lemma unbox_known_named_free (names : list variable) (known : known_numbers)
    (d : named) :
  numbers_within names known ->
  incl (free_variables_of_named (unbox_known_named known d))
    (free_variables_of_named d ++ names).
Proof.
  intros Hknown. unfold unbox_known_named.
  destruct d as [ | p args | | | | | ]; try (apply incl_appl, incl_refl).
  destruct (unwrapping_of_known known p args) as [contents | ]
    eqn:Hunwrapping; [ | apply incl_appl, incl_refl ].
  cbn [free_variables_of_named]. apply incl_appr.
  unfold unwrapping_of_known in Hunwrapping.
  destruct (unary_applied p args) as [[p' arg] | ]; [ | discriminate ].
  destruct p'; try discriminate;
    exact (unwrapped_free names known _ _ _ Hknown Hunwrapping).
Qed.

Fixpoint unbox_numbers_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : known_numbers),
    numbers_within names known ->
    scoped_under names e (unbox_numbers_knowing known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown;
    cbn [unbox_numbers_knowing unbox_numbers_in_handler];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply unbox_known_named_free, Hknown | ].
    apply unbox_numbers_knowing_scoped, numbers_under_let_within, Hknown.
  - apply scoped_under_let_static. apply unbox_numbers_knowing_scoped.
    intros entry Hentry.
    eapply incl_tran; [ exact (Hknown entry Hentry) | ].
    apply incl_appl, incl_refl.
  - apply scoped_under_let_cont; apply unbox_numbers_knowing_scoped;
      [ apply forget_wrapped_vars_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply unbox_numbers_knowing_scoped, Hknown.
Qed.

Lemma unbox_known_numbers_scoped (e : expr) :
  scoped e (unbox_known_numbers e).
Proof.
  apply scoped_under_none, unbox_numbers_knowing_scoped. intros entry [].
Qed.

Theorem unbox_known_numbers_keeps_scope :
  unbox_known_numbers_keeps_scope_statement.
Proof. apply keeps_scope_intro, unbox_known_numbers_scoped. Qed.

(* Sharing the primitives already computed. *)

Definition computed_within (names : list variable) (known : computed)
    : Prop :=
  ∀ entry : variable * (prim * list simple),
    In entry known -> In (fst entry) names.

Lemma forget_computed_within (names names' : list variable) (x : variable)
    (known : computed) :
  incl names names' ->
  computed_within names known ->
  computed_within names' (forget_computed x known).
Proof.
  intros Hincl Hknown entry Hentry. unfold forget_computed in Hentry.
  apply filter_In in Hentry. exact (Hincl _ (Hknown entry (proj1 Hentry))).
Qed.

Lemma forget_computed_vars_within (names : list variable)
    (xs : list variable) :
  ∀ known : computed,
    computed_within names known ->
    computed_within names (forget_computed_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known Hknown;
    cbn [forget_computed_vars]; [ exact Hknown | ].
  apply IH, (forget_computed_within names names), Hknown. apply incl_refl.
Qed.

Lemma computed_under_let_within (names : list variable) (known : computed)
    (x : variable) (d : named) :
  computed_within names known ->
  computed_within (x :: names ++ free_variables_of_named d)
    (computed_under_let known x d).
Proof.
  intros Hknown. unfold computed_under_let. cbv zeta.
  assert (Hforgot :
    computed_within (x :: names ++ free_variables_of_named d)
      (forget_computed x known))
    by (apply (forget_computed_within names); [ | exact Hknown ];
        apply incl_tl, incl_appl, incl_refl).
  destruct d as [ | p args | | | | | ]; try exact Hforgot.
  destruct (shareable p && negb (var_in_simples x args)); [ | exact Hforgot ].
  intros entry [<- | Hentry]; [ left; reflexivity | ].
  apply in_app_or in Hentry as [Hagain | Hentry];
    [ | exact (Hforgot entry Hentry) ].
  right. apply in_or_app. right.
  exact (FlambdaCseProofs.tagged_again_free x p args entry Hagain).
Qed.

Lemma computed_by_from (known : computed) (p : prim) (args : list simple)
    (x : variable) :
  computed_by known p args = Some x ->
  ∃ entry : variable * (prim * list simple), In entry known ∧ fst entry = x.
Proof.
  induction known as [ | [y [p' args']] known IH]; cbn [computed_by];
    [ discriminate | ].
  destruct (prim_eqb p p' && same_simples args args').
  - intros Hfound. injection Hfound as ->.
    exists (x, (p', args')). split; [ left; reflexivity | reflexivity ].
  - intros Hfound. destruct (IH Hfound) as (entry & Hentry & Hfst).
    exists entry. split; [ right; exact Hentry | exact Hfst ].
Qed.

(* What the sharing knows in the body of a [Let]: the array that the
   [Let] makes is known of its variable. *)
Lemma computed_or_made_under_let_within (names : list variable)
    (known : computed) (x : variable) (d : named) :
  computed_within names known ->
  computed_within (x :: names ++ free_variables_of_named d)
    (computed_or_made_under_let known x d).
Proof.
  intros Hknown entry Hentry. unfold computed_or_made_under_let in Hentry.
  apply in_app_or in Hentry as [Hmade | Hentry];
    [ | exact (computed_under_let_within names known x d Hknown entry Hentry) ].
  unfold array_made_under_let in Hmade.
  destruct d as [ | p args | | | | | ]; try contradiction.
  destruct p as [ | | | hprim | ]; try contradiction.
  destruct hprim as [ | | kind | ]; try contradiction.
  destruct (var_in_simples x args); [ contradiction | ].
  destruct Hmade as [<- | []]. left. reflexivity.
Qed.

(* The length of an array made becomes a constant, which has no free
   variable. *)
Lemma made_length_named_closed (known : computed) (d d' : named) :
  made_length_named known d = Some d' -> free_variables_of_named d' = [].
Proof. unfold made_length_named. intros Hmade. tests. reflexivity. Qed.

Lemma share_named_free (names : list variable) (known : computed)
    (d : named) :
  computed_within names known ->
  incl (free_variables_of_named (share_named known d))
    (free_variables_of_named d ++ names).
Proof.
  intros Hknown. unfold share_named.
  destruct (made_length_named known d) as [d' | ] eqn:Hlength;
    [ rewrite (made_length_named_closed known d d' Hlength);
      apply incl_nil_l
    | ].
  destruct d as [ | p args | | | | | ]; try (apply incl_appl, incl_refl).
  destruct (shareable p); [ | apply incl_appl, incl_refl ].
  destruct (computed_by known p args) as [x | ] eqn:Hcomputed;
    [ | apply incl_appl, incl_refl ].
  destruct (computed_by_from known p args x Hcomputed)
    as (entry & Hentry & <-).
  cbn [free_variables_of_named free_variables_of_simple].
  intros y [<- | []]. apply in_or_app. right. exact (Hknown entry Hentry).
Qed.

Fixpoint share_primitives_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : computed),
    computed_within names known ->
    scoped_under names e (share_primitives_knowing known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown;
    cbn [share_primitives_knowing share_primitives_in_handler];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply share_named_free, Hknown | ].
    apply share_primitives_knowing_scoped,
      computed_or_made_under_let_within, Hknown.
  - apply scoped_under_let_static. apply share_primitives_knowing_scoped.
    intros entry Hentry. apply in_or_app. left. exact (Hknown entry Hentry).
  - apply scoped_under_let_cont; apply share_primitives_knowing_scoped;
      [ apply forget_computed_vars_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply share_primitives_knowing_scoped, Hknown.
Qed.

Lemma share_known_primitives_scoped (e : expr) :
  scoped e (share_known_primitives e).
Proof.
  apply scoped_under_none, share_primitives_knowing_scoped. intros entry [].
Qed.

Theorem share_known_primitives_keeps_scope :
  share_known_primitives_keeps_scope_statement.
Proof. apply keeps_scope_intro, share_known_primitives_scoped. Qed.

(* The length of an array that a [Let] in scope has made, alone: a
   [Let] of a length becomes a [Let] of a constant, and nothing else
   moves. *)
Fixpoint lengths_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : computed),
    scoped_under names e (lengths_knowing known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known;
    cbn [lengths_knowing lengths_in_handler];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ | apply lengths_knowing_scoped ].
    destruct (made_length_named known d) as [d' | ] eqn:Hlength;
      [ rewrite (made_length_named_closed known d d' Hlength) | ]; names.
  - apply scoped_under_let_static. apply lengths_knowing_scoped.
  - apply scoped_under_let_cont; apply lengths_knowing_scoped.
  - apply scoped_under_let_cont_recursive. apply lengths_knowing_scoped.
Qed.

Lemma fold_made_lengths_scoped (e : expr) : scoped e (fold_made_lengths e).
Proof. apply scoped_under_none, lengths_knowing_scoped. Qed.

Theorem fold_made_lengths_keeps_scope :
  fold_made_lengths_keeps_scope_statement.
Proof. apply keeps_scope_intro, fold_made_lengths_scoped. Qed.

(* Following aliases. *)

Definition type_within (names : list variable) (t : flambda_type) : Prop :=
  match t with
  | Equals s => incl (free_variables_of_simple s) names
  | Closure_of _ slots => incl (free_variables_of_value_slots slots) names
  | Unknown | Bottom => True
  end.

Definition tenv_within (names : list variable) (tenv : typing_env) : Prop :=
  ∀ entry : variable * flambda_type,
    In entry tenv -> type_within names (snd entry).

Lemma tenv_within_weaken (names names' : list variable) (tenv : typing_env) :
  incl names names' -> tenv_within names tenv -> tenv_within names' tenv.
Proof.
  intros Hincl Htenv entry Hentry. pose proof (Htenv entry Hentry) as Htype.
  destruct (snd entry); cbn [type_within] in Htype |- *; try exact I;
    eapply incl_tran; eassumption.
Qed.

Lemma drop_var_within (names : list variable) (x : variable)
    (tenv : typing_env) :
  tenv_within names tenv -> tenv_within names (drop_var x tenv).
Proof.
  intros Htenv entry Hentry. unfold drop_var in Hentry.
  apply filter_In in Hentry. exact (Htenv entry (proj1 Hentry)).
Qed.

Lemma drop_vars_within (names : list variable) (xs : list variable)
    (tenv : typing_env) :
  tenv_within names tenv -> tenv_within names (drop_vars xs tenv).
Proof.
  intros Htenv. unfold drop_vars.
  induction xs as [ | x xs IH]; cbn [fold_right];
    [ exact Htenv | apply drop_var_within, IH ].
Qed.

Lemma find_alias_from (tenv : typing_env) (x : variable) (s : simple) :
  find_alias tenv x = Some s -> ∃ y : variable, In (y, Equals s) tenv.
Proof.
  induction tenv as [ | [y t] tenv IH]; cbn [find_alias]; [ discriminate | ].
  destruct t as [ | | s' | code slots];
    try (intros Hfound; destruct (IH Hfound) as [y' Hin]; exists y'; right;
         exact Hin).
  destruct (Pos.eqb x y).
  - intros Hfound. injection Hfound as ->. exists y. left. reflexivity.
  - intros Hfound. destruct (IH Hfound) as [y' Hin]. exists y'. right.
    exact Hin.
Qed.

Lemma find_closure_from (tenv : typing_env) (x : variable) (code : code_id)
    (slots : list (value_slot * simple)) :
  find_closure tenv x = Some (code, slots) ->
  ∃ y : variable, In (y, Closure_of code slots) tenv.
Proof.
  induction tenv as [ | [y t] tenv IH]; cbn [find_closure];
    [ discriminate | ].
  destruct t as [ | | s' | code' slots'];
    try (intros Hfound; destruct (IH Hfound) as [y' Hin]; exists y'; right;
         exact Hin).
  destruct (Pos.eqb x y).
  - intros Hfound. injection Hfound as -> ->. exists y. left. reflexivity.
  - intros Hfound. destruct (IH Hfound) as [y' Hin]. exists y'. right.
    exact Hin.
Qed.

Section Canonical.

Variables (names : list variable) (tenv : typing_env).

Hypothesis within : tenv_within names tenv.

Lemma canonical_simple_free (s : simple) :
  incl (free_variables_of_simple (canonical_simple tenv s))
    (free_variables_of_simple s ++ names).
Proof.
  unfold canonical_simple.
  destruct s as [x | sym | c]; try (apply incl_appl, incl_refl).
  destruct (find_alias tenv x) as [alias | ] eqn:Halias;
    [ | apply incl_appl, incl_refl ].
  destruct (find_alias_from tenv x alias Halias) as [y Hin].
  apply incl_appr. exact (within (y, Equals alias) Hin).
Qed.

Lemma canonical_simples_free (args : list simple) :
  incl (free_variables_of_simples (map (canonical_simple tenv) args))
    (free_variables_of_simples args ++ names).
Proof. apply incl_flat_map_under, canonical_simple_free. Qed.

Lemma canonical_value_slots_free (slots : list (value_slot * simple)) :
  incl (free_variables_of_value_slots (canonical_value_slots tenv slots))
    (free_variables_of_value_slots slots ++ names).
Proof.
  apply
    (incl_flat_map_under (fun slot => free_variables_of_simple (snd slot))
       (fun slot => free_variables_of_simple (snd slot))).
  intros slot. cbn [snd]. apply canonical_simple_free.
Qed.

Lemma canonical_project_free (slot : value_slot) (closure : simple) :
  incl (free_variables_of_named (canonical_project tenv slot closure))
    (free_variables_of_simple closure ++ names).
Proof.
  unfold canonical_project.
  destruct (closure_of_simple tenv closure) as [[code slots] | ] eqn:Hclosure;
    [ | apply incl_appl, incl_refl ].
  destruct (lookup slot slots) as [contents | ] eqn:Hslot;
    [ | apply incl_appl, incl_refl ].
  cbn [free_variables_of_named]. apply incl_appr.
  destruct closure as [x | sym | c]; cbn [closure_of_simple] in Hclosure;
    try discriminate.
  destruct (find_closure_from tenv x code slots Hclosure) as [y Hin].
  pose proof (within (y, Closure_of code slots) Hin) as Hslots.
  cbn [snd type_within] in Hslots.
  eapply incl_tran; [ | exact Hslots ]. intros a Ha.
  unfold free_variables_of_value_slots. apply in_flat_map.
  exists (slot, contents). split; [ exact (lookup_in _ _ _ Hslot) | exact Ha ].
Qed.

Lemma canonical_named_free (d : named) :
  incl (free_variables_of_named (canonical_named tenv d))
    (free_variables_of_named d ++ names).
Proof.
  destruct d; cbn [canonical_named free_variables_of_named];
    try
      first
        [ apply canonical_simple_free | apply canonical_simples_free
        | apply canonical_value_slots_free ].
  eapply incl_tran; [ apply canonical_project_free | ].
  intros y Hy. apply in_app_or in Hy.
  destruct Hy as [Hy | Hy];
    [ exact (canonical_simple_free _ y Hy) | apply in_or_app; right; exact Hy ].
Qed.

Lemma canonical_static_const_free (const : static_const) :
  incl
    (free_variables_of_static_const (canonical_static_const tenv const))
    (free_variables_of_static_const const ++ names).
Proof.
  destruct const; cbn [canonical_static_const free_variables_of_static_const];
    first
      [ apply canonical_simple_free | apply canonical_simples_free
      | apply canonical_value_slots_free | apply incl_appl, incl_refl ].
Qed.

Lemma canonical_static_bindings_scope (bindings : list static_binding) :
  (forallb binding_code_closed bindings = true ->
   forallb binding_code_closed (map (canonical_static_binding tenv) bindings)
   = true)
  ∧ incl
      (flat_map free_variables_of_binding
         (map (canonical_static_binding tenv) bindings))
      (flat_map free_variables_of_binding bindings ++ names)
  ∧ incl
      (flat_map free_continuations_of_binding
         (map (canonical_static_binding tenv) bindings))
      (flat_map free_continuations_of_binding bindings).
Proof.
  split; [ | split ].
  - induction bindings as [ | binding bindings IH]; cbn [map forallb];
      [ intros _; reflexivity | ].
    intros Hclosed. apply andb_true_iff in Hclosed.
    destruct Hclosed as [Hfirst Hrest]. rewrite (IH Hrest), andb_true_r.
    destruct binding; exact Hfirst.
  - apply incl_flat_map_under. intros binding.
    destruct binding as [id c | id | sym const];
      cbn [canonical_static_binding free_variables_of_binding];
      [ apply incl_appl, incl_refl | apply incl_appl, incl_refl
      | apply canonical_static_const_free ].
  - rewrite
      (flat_map_map_same free_continuations_of_binding
         free_continuations_of_binding);
      [ apply incl_refl | ].
    intros binding. destruct binding; reflexivity.
Qed.

Lemma canonical_apply_cont_free (ac : apply_cont) :
  incl (free_variables_of_apply_cont (canonical_apply_cont tenv ac))
    (free_variables_of_apply_cont ac ++ names).
Proof.
  unfold free_variables_of_apply_cont, canonical_apply_cont. cbn [ac_args].
  apply canonical_simples_free.
Qed.

Lemma exn_extra_simples_canonical (ap : apply_expr) :
  exn_extra_simples (canonical_apply tenv ap)
  = map (canonical_simple tenv) (exn_extra_simples ap).
Proof.
  unfold exn_extra_simples, canonical_apply. cbn [ap_exn_extra_args].
  rewrite !map_map. reflexivity.
Qed.

Lemma canonical_apply_free (ap : apply_expr) :
  incl (free_variables_of_apply (canonical_apply tenv ap))
    (free_variables_of_apply ap ++ names).
Proof.
  unfold free_variables_of_apply. rewrite exn_extra_simples_canonical.
  unfold canonical_apply. cbn [ap_callee ap_args].
  apply incl_app_under; [ apply canonical_simple_free | ].
  apply incl_app_under; apply canonical_simples_free.
Qed.

End Canonical.

Lemma typing_env_under_let_within (names : list variable)
    (tenv : typing_env) (x : variable) (d : named) :
  tenv_within names tenv ->
  tenv_within (names ++ free_variables_of_named d)
    (typing_env_under_let tenv x d).
Proof.
  intros Htenv.
  assert (Hdropped :
    tenv_within (names ++ free_variables_of_named d) (drop_var x tenv)).
  { apply drop_var_within.
    eapply tenv_within_weaken; [ | exact Htenv ]. apply incl_appl, incl_refl. }
  unfold typing_env_under_let.
  destruct d as [s | p args | code slots | | | | ]; try exact Hdropped.
  - destruct (var_in_simple x s); [ exact Hdropped | ].
    intros entry [<- | Hentry]; [ | exact (Hdropped entry Hentry) ].
    cbn [snd type_within free_variables_of_named]. apply incl_appr, incl_refl.
  - (* What is known of the result of a store is a constant, which
       mentions no variable. *)
    destruct p as [u | b | vr | q | r]; try exact Hdropped.
    destruct q as [mut tag | kind init field | kind | kind set_kind];
      try exact Hdropped;
      (intros entry [<- | Hentry]; [ | exact (Hdropped entry Hentry) ]);
      intros y [].
  - destruct (var_in_value_slots x slots); [ exact Hdropped | ].
    intros entry [<- | Hentry]; [ | exact (Hdropped entry Hentry) ].
    cbn [snd type_within free_variables_of_named]. apply incl_appr, incl_refl.
Qed.

Fixpoint follow_aliases_scoped_under (e : expr) :
  ∀ (names : list variable) (tenv : typing_env),
    tenv_within names tenv -> scoped_under names e (follow_aliases tenv e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names tenv Htenv; cbn [follow_aliases follow_aliases_in_handler].
  - apply scoped_under_let; [ apply canonical_named_free, Htenv | ].
    apply follow_aliases_scoped_under.
    eapply tenv_within_weaken; [ | apply typing_env_under_let_within, Htenv ].
    intros y Hy. apply in_app_or in Hy. right. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ left; exact Hy | ].
    apply (canonical_named_free names tenv Htenv d), in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ right; exact Hy | left; exact Hy ].
  - destruct (canonical_static_bindings_scope names tenv Htenv bindings)
      as (Hcode & Hv & Hk).
    apply scoped_under_let_static_bindings;
      [ exact Hcode | exact Hv | exact Hk | ].
    apply follow_aliases_scoped_under.
    eapply tenv_within_weaken; [ | exact Htenv ]. apply incl_appl, incl_refl.
  - apply scoped_under_let_cont; apply follow_aliases_scoped_under;
      [ apply drop_vars_within, Htenv | exact Htenv ].
  - apply scoped_under_let_cont_recursive.
    apply follow_aliases_scoped_under, Htenv.
  - intros _. scope.
    split; [ reflexivity | split; [ | apply incl_refl ] ].
    apply canonical_apply_free, Htenv.
  - intros _. scope.
    split; [ reflexivity | split; [ | apply incl_refl ] ].
    apply canonical_apply_cont_free, Htenv.
  - intros _. scope. split; [ reflexivity | split ].
    + apply incl_app_under; [ apply canonical_simple_free, Htenv | ].
      apply
        (incl_flat_map_under
           (fun arm => free_variables_of_apply_cont (snd arm))
           (fun arm => free_variables_of_apply_cont (snd arm))).
      intros arm. cbn [snd]. apply canonical_apply_cont_free, Htenv.
    + rewrite
        (flat_map_map_same (fun arm => apply_cont_continuations (snd arm))
           (fun arm => apply_cont_continuations (snd arm)));
        [ apply incl_refl | intros arm; reflexivity ].
  - apply scoped_under_refl.
Qed.

Lemma follow_aliases_scoped (e : expr) : scoped e (follow_aliases [] e).
Proof.
  apply scoped_under_none, follow_aliases_scoped_under. intros entry [].
Qed.

Theorem follow_aliases_keeps_scope : follow_aliases_keeps_scope_statement.
Proof. apply keeps_scope_intro, follow_aliases_scoped. Qed.
