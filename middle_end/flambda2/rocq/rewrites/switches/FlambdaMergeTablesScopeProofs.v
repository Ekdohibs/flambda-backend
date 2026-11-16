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

(* Proof of the statement of FlambdaMergeTablesScopeSpec.v. Nothing
   here needs review. It follows the proof for the merging of the arms
   of a [Switch] in FlambdaKeepsScopeProofs.v, whose lemmas about the
   [Let]s that bind the variable of the scrutinee again, and about
   those that bind another variable to something computed of it, it
   uses. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaNormaliseSpec
  FlambdaMergeSwitchesSpec FlambdaValidatorEquivarianceSpec
  FlambdaScopedProofs FlambdaKeepsScopeProofs FlambdaMergeTablesSpec
  FlambdaMergeTablesScopeSpec.
Import ListNotations.

(* The [Let]s bind the variable of the scrutinee again, and read it and
   the symbol of the table. *)
Lemma bind_table_offset_binds_again (v : variable) (first offset : Z) :
  binds_again v (bind_table_offset v first offset).
Proof.
  intros body. unfold bind_table_offset.
  destruct (first =? 0)%Z; [ exact (binds_again_nothing v body) | ].
  split; [ reflexivity | split; [ | reflexivity ] ].
  scope. simpl.
  intros x [<- | Hx]; [ left; reflexivity | right ].
  apply in_without in Hx. apply Hx.
Qed.

Lemma table_load_binds_again (v : variable) (s : symbol) (immediates : bool) :
  binds_again v
    (fun body =>
       Let v Value
         (Prim
            (Binary
               (Array_load (table_array_kind immediates)
                  (table_load_kind immediates) Immutable))
            [Symbol s; Var v])
         body).
Proof.
  intros body. split; [ reflexivity | split; [ | reflexivity ] ].
  scope. simpl.
  intros x [<- | Hx]; [ left; reflexivity | right ].
  apply in_without in Hx. apply Hx.
Qed.

Lemma bind_table_load_binds_again (v : variable) (s : symbol)
    (first offset : Z) (immediates : bool) :
  binds_again v (bind_table_load v s first offset immediates).
Proof.
  exact
    (binds_again_twice v _ _ (bind_tagged_binds_again v)
       (binds_again_twice v _ _ (bind_table_offset_binds_again v first offset)
          (table_load_binds_again v s immediates))).
Qed.

(* The merged arguments are among those of the first arm and the
   variable of the scrutinee. *)
Lemma merged_across_free (v : variable) (arms : list (Z * apply_cont)) :
  ∀ args : list simple,
    incl (free_variables_of_simples (merged_across v args arms))
      (v :: free_variables_of_simples args).
Proof.
  induction arms as [ | arm arms IH]; intros args; cbn [merged_across].
  - apply incl_tl, incl_refl.
  - intros x Hx. apply IH in Hx.
    destruct Hx as [<- | Hx]; [ left; reflexivity | ].
    exact (merged_arguments_free v args _ x Hx).
Qed.

Lemma merge_switch_by_table_scoped (tables : known_tables)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (merged : expr) :
  merge_switch_by_table tables scrutinee arms = Some merged ->
  scoped (Switch scrutinee arms) merged.
Proof.
  unfold merge_switch_by_table.
  destruct scrutinee as [v | sym | c]; try discriminate.
  destruct arms as [ | [first ac] [ | arm1 [ | arm2 others]]];
    try discriminate.
  cbv zeta.
  destruct (var_in_simples v _); [ | discriminate ].
  destruct (find _ tables) as [[s elements] | ]; [ | discriminate ].
  intros [= <-] _.
  change
    (merged_across v
       (merged_arguments v
          (merged_arguments v (ac_args ac) (ac_args (snd arm1)))
          (ac_args (snd arm2)))
       others)
    with (merged_across v (ac_args ac) (arm1 :: arm2 :: others)).
  destruct
    (bind_table_load_binds_again v s first (table_offset first)
       (all_immediates elements)
       (Apply_cont
          (Apply_cont_expr (ac_continuation ac)
             (merged_across v (ac_args ac) (arm1 :: arm2 :: others)) None)))
    as (Hclosed & Hv & Hk).
  rewrite Hclosed, Hk. split; [ reflexivity | split ].
  - assert
      (Hswitch :
        ∀ y : variable,
          y = v ∨ In y (free_variables_of_simples (ac_args ac)) ->
          In y
            (free_variables_of_expr
               (Switch (Var v) ((first, ac) :: arm1 :: arm2 :: others)))).
    { intros y [-> | Hy]; scope; simpl;
        [ left; reflexivity | right; apply in_or_app; left; exact Hy ]. }
    intros x Hx. apply Hswitch. apply Hv in Hx.
    destruct Hx as [<- | Hx]; [ left; reflexivity | ].
    scope_in Hx. apply merged_across_free in Hx.
    destruct Hx as [<- | Hx]; [ left; reflexivity | right; exact Hx ].
  - scope. simpl. intros k [<- | []]. left. reflexivity.
Qed.

(* A way for each argument. The [Let]s of a load bind a variable to
   something computed of the scrutinee, which is not bound again, and
   read the symbol of the table. *)
Lemma bind_table_load_to_binds_from (v x : variable) (s : symbol)
    (first offset : Z) (immediates : bool) :
  binds_from v x (bind_table_load_to v x s first offset immediates).
Proof.
  exact
    (binds_from_then v x _ _ (bind_tagged_to_binds_from v x)
       (binds_again_twice x _ _ (bind_table_offset_binds_again x first offset)
          (table_load_binds_again x s immediates))).
Qed.

(* The names that the [Let]s of the ways bind. *)
Fixpoint table_names_bound (names : list variable) (ways : list table_way)
    : list variable :=
  match names, ways with
  | x :: names, way :: ways =>
    match way with
    | Without_table (Same_in_every_arm _ | Computed As_it_is) =>
      table_names_bound names ways
    | Without_table
        (Computed
           (By_tagging | By_tagging_and_negating | By_arithmetic _ _ _))
    | From_table _ _ _ =>
      x :: table_names_bound names ways
    end
  | _, _ => []
  end.

Lemma bind_table_ways_scope (first offset : Z) (v : variable) (body : expr)
    (names : list variable) :
  ∀ ways : list table_way,
    code_closed (bind_table_ways first offset v names ways body)
    = code_closed body
    ∧ (∀ y : variable,
         In y
           (free_variables_of_expr
              (bind_table_ways first offset v names ways body)) ->
         y = v
         ∨ (In y (free_variables_of_expr body)
            ∧ ~ In y (table_names_bound names ways)))
    ∧ free_continuations_of_expr
        (bind_table_ways first offset v names ways body)
      = free_continuations_of_expr body.
Proof.
  induction names as [ | x names IH]; intros [ | way ways];
    cbn [bind_table_ways table_names_bound];
    try
      (split; [ reflexivity | split; [ | reflexivity ] ];
       intros y Hy; right; split; [ exact Hy | intros [] ]).
  destruct (IH ways) as (Hclosed & Hv & Hk).
  set (rest := bind_table_ways first offset v names ways body) in *.
  assert
    (Hbinds :
      ∀ wrap : expr -> expr,
        binds_from v x wrap ->
        code_closed (wrap rest) = code_closed body
        ∧ (∀ y : variable,
             In y (free_variables_of_expr (wrap rest)) ->
             y = v
             ∨ (In y (free_variables_of_expr body)
                ∧ ~ In y (x :: table_names_bound names ways)))
        ∧ free_continuations_of_expr (wrap rest)
          = free_continuations_of_expr body).
  { intros wrap Hwrap.
    destruct (Hwrap rest) as (Hclosed' & Hv' & Hk').
    split; [ congruence | split; [ | congruence ] ].
    intros y Hy. apply Hv' in Hy.
    destruct Hy as [-> | [Hy Hne]]; [ left; reflexivity | ].
    apply Hv in Hy.
    destruct Hy as [-> | [Hy Hbound]]; [ left; reflexivity | right ].
    split; [ exact Hy | ].
    intros [Heq | Hin]; [ apply Hne; symmetry; exact Heq | exact (Hbound Hin) ].
  }
  destruct way as [[s | how] | s elements immediates]; cbn [bind_table_way].
  - split; [ exact Hclosed | split; [ exact Hv | exact Hk ] ].
  - destruct how as [ | | | kind offset' slope].
    + apply (Hbinds (bind_way By_tagging v x)), bind_way_binds_from.
      discriminate.
    + apply (Hbinds (bind_way By_tagging_and_negating v x)),
        bind_way_binds_from.
      discriminate.
    + split; [ exact Hclosed | split; [ exact Hv | exact Hk ] ].
    + apply (Hbinds (bind_way (By_arithmetic kind offset' slope) v x)),
        bind_way_binds_from.
      discriminate.
  - exact
      (Hbinds _ (bind_table_load_to_binds_from v x s first offset immediates)).
Qed.

(* The merged arguments are the scrutinee, the names that are bound,
   and what the arguments of an arm mention. *)
Lemma merged_by_table_ways_free (first offset : Z) (v : variable) (i : Z)
    (names : list variable) :
  ∀ (ways : list table_way) (args : list simple),
    arguments_by_table_ways first offset i ways = Some args ->
    ∀ y : variable,
      In y (free_variables_of_simples (merged_by_table_ways v names ways)) ->
      y = v
      ∨ In y (table_names_bound names ways)
      ∨ In y (free_variables_of_simples args).
Proof.
  induction names as [ | x names IH]; intros [ | way ways] args Hargs y Hy;
    cbn [merged_by_table_ways] in Hy; try (destruct Hy).
  cbn [arguments_by_table_ways] in Hargs.
  destruct (argument_by_table_way first offset i way) as [s0 | ] eqn:Hs0;
    [ | discriminate ].
  destruct (arguments_by_table_ways first offset i ways) as [args0 | ]
    eqn:Hargs0; [ | discriminate ].
  injection Hargs as <-.
  unfold free_variables_of_simples in Hy. cbn [flat_map] in Hy.
  apply in_app_or in Hy.
  unfold free_variables_of_simples. cbn [flat_map table_names_bound].
  destruct Hy as [Hy | Hy].
  - destruct way as [[s | how] | s elements immediates];
      cbn [argument_by_table_way argument_in_arm] in Hs0.
    + injection Hs0 as <-. right. right. apply in_or_app. left. exact Hy.
    + destruct how; simpl in Hy; destruct Hy as [<- | []];
        first [ left; reflexivity | right; left; left; reflexivity ].
    + simpl in Hy. destruct Hy as [<- | []]. right. left. left. reflexivity.
  - destruct (IH ways args0 Hargs0 y Hy) as [-> | [Hin | Hin]].
    + left. reflexivity.
    + right. left.
      destruct way as
        [[s | [ | | | kind offset' slope]] | s elements immediates];
        first [ exact Hin | right; exact Hin ].
    + right. right. apply in_or_app. right. exact Hin.
Qed.

Lemma merge_switch_by_table_ways_scoped (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (merged : expr) :
  merge_switch_by_table_ways known tables scrutinee arms = Some merged ->
  scoped (Switch scrutinee arms) merged.
Proof.
  unfold merge_switch_by_table_ways.
  destruct scrutinee as [v | sym | c]; try discriminate.
  destruct arms as [ | [first [k0 args trap]] [ | arm1 [ | arm2 others]]];
    try discriminate.
  cbv zeta. cbn [ac_continuation ac_args].
  destruct (lookup k0 known) as [names | ]; [ | discriminate ].
  destruct (table_ways_of_arguments _ _ _ _ _) as [ways | ];
    [ | discriminate ].
  destruct (existsb is_from_table ways && _ && _) eqn:Hok; [ | discriminate ].
  intros [= <-] _.
  apply andb_prop in Hok as [_ Hall].
  cbn [forallb] in Hall. apply andb_prop in Hall as [Hfirst _].
  unfold arm_has_table_ways in Hfirst. cbn [fst snd] in Hfirst.
  destruct (arguments_by_table_ways first (table_offset first) first ways)
    as [args0 | ] eqn:Hargs0; [ | discriminate ].
  destruct (apply_cont_eq_dec _ _) as [Heq | ]; [ | discriminate ].
  injection Heq as -> ->.
  destruct
    (bind_table_ways_scope first (table_offset first) v
       (Apply_cont
          (Apply_cont_expr k0 (merged_by_table_ways v names ways) None))
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
                  ((first, Apply_cont_expr k0 args0 None)
                   :: arm1 :: arm2 :: others)))).
    { intros y [-> | Hy]; scope; simpl;
        [ left; reflexivity | right; apply in_or_app; left; exact Hy ]. }
    intros y Hy. apply Hswitch. apply Hv in Hy.
    destruct Hy as [-> | [Hy Hbound]]; [ left; reflexivity | ].
    scope_in Hy.
    destruct
      (merged_by_table_ways_free first (table_offset first) v first names
         ways args0 Hargs0 y Hy)
      as [-> | [Hin | Hin]];
      [ left; reflexivity | contradiction | right; exact Hin ].
  - scope. simpl. intros k [<- | []]. left. reflexivity.
Qed.

Lemma merge_switch_with_tables_scoped (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  scoped (Switch scrutinee arms)
    (merge_switch_with_tables known tables scrutinee arms).
Proof.
  unfold merge_switch_with_tables.
  destruct tables as [ | table tables]; [ apply scoped_refl | ].
  destruct (merge_switch_by_table (table :: tables) scrutinee arms)
    as [merged | ] eqn:Hmerged.
  - destruct (merge_switch scrutinee arms); try apply scoped_refl.
    exact
      (merge_switch_by_table_scoped (table :: tables) scrutinee arms merged
         Hmerged).
  - destruct
      (merge_switch_by_table_ways known (table :: tables) scrutinee arms)
      as [merged | ] eqn:Hways; [ | apply scoped_refl ].
    exact
      (merge_switch_by_table_ways_scoped known (table :: tables) scrutinee
         arms merged Hways).
Qed.

Fixpoint merge_tables_under_scoped (e : expr) :
  ∀ (known : known_continuations) (tables : known_tables),
    scoped e (merge_tables_under known tables e)
with merge_tables_in_handler_scoped (h : continuation_handler) :
  ∀ (known : known_continuations) (tables : known_tables),
    scoped_handler h (merge_tables_in_handler known tables h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known tables;
      cbn [merge_tables_under merge_tables_in_handler].
    + apply scoped_let;
        [ apply incl_refl | apply merge_tables_under_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply merge_tables_under_scoped ].
    + apply scoped_let_cont;
        [ apply merge_tables_in_handler_scoped
        | apply merge_tables_under_scoped ].
    + cbv zeta.
      generalize
        (map (fun entry => (fst entry, parameters_of_handler (snd entry)))
           handlers
         ++ known).
      intros known'.
      apply scoped_let_cont_recursive;
        [ | | apply merge_tables_under_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply merge_tables_in_handler_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply merge_switch_with_tables_scoped.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros known tables.
    cbn [merge_tables_under merge_tables_in_handler].
    apply scoped_handler_intro, merge_tables_under_scoped.
Qed.

Lemma merge_tables_knowing_scoped (e : expr) (tables : known_tables) :
  scoped e (merge_tables_knowing tables e).
Proof. apply merge_tables_under_scoped. Qed.

(* In the bodies of code it is a rewrite taken there. *)
Fixpoint merge_tables_in_code_scoped (e : expr) :
  ∀ tables : known_tables, scoped e (merge_tables_in_code tables e)
with merge_tables_in_code_in_handler_scoped (h : continuation_handler) :
  ∀ tables : known_tables,
    scoped_handler h (merge_tables_in_code_in_handler tables h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros tables;
      cbn [merge_tables_in_code merge_tables_in_code_in_handler];
      cbv zeta.
    + apply scoped_let;
        [ apply incl_refl | apply merge_tables_in_code_scoped ].
    + apply scoped_let_static;
        [ apply rewrite_code_in_bindings_scoped, keeps_scope_intro;
          intros code_body; apply merge_tables_knowing_scoped
        | apply merge_tables_in_code_scoped ].
    + apply scoped_let_cont;
        [ apply merge_tables_in_code_in_handler_scoped
        | apply merge_tables_in_code_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply merge_tables_in_code_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply merge_tables_in_code_in_handler_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros tables.
    cbn [merge_tables_in_code merge_tables_in_code_in_handler].
    apply scoped_handler_intro, merge_tables_in_code_scoped.
Qed.

Lemma merge_tables_scoped (e : expr) : scoped e (merge_tables e).
Proof.
  unfold merge_tables.
  eapply scoped_trans;
    [ apply merge_tables_knowing_scoped | apply merge_tables_in_code_scoped ].
Qed.

Theorem merge_tables_keeps_scope : merge_tables_keeps_scope_statement.
Proof. apply keeps_scope_intro, merge_tables_scoped. Qed.
