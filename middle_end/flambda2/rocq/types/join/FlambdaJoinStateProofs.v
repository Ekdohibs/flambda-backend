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


(* Proofs of the statements of FlambdaJoinStateSpec.v. Nothing in this
   file needs to be reviewed. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaJoinEnv FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec.
Import ListNotations.

(* ---------------------------------------------------------------------- *)
(* The valuation of the target                                             *)
(* ---------------------------------------------------------------------- *)

Lemma lookup_target_valuation (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) (x : variable) :
  lookup x (target_valuation s index use valuation)
  = match value_of_definition s index valuation x with
    | Some v => Some v
    | None => if known_to s use x then lookup x valuation else None
    end.
Proof.
  unfold target_valuation.
  rewrite lookup_app, lookup_tabulate, among_domain, lookup_filter_bindings.
  unfold value_of_definition, is_bound.
  destruct (lookup x (Join_env.definitions_of_existentials s))
    as [[simples kind] | ].
  - destruct (match Simples_in_joined_envs.find index simples with
              | Some simple => eval_simple valuation simple
              | None => None
              end); [ reflexivity | ].
    destruct (lookup x valuation); destruct (known_to s use x); reflexivity.
  - destruct (lookup x valuation); destruct (known_to s use x); reflexivity.
Qed.

(* A variable that an input or the source knows has no definition. *)
Lemma known_no_definition (s : Join_env.t) (index index' : Index.t)
    (use : Typing_env.t) (valuation : env) (x : variable) :
  state_valid s ->
  Join_env.joined_env s index = Some use ->
  known_to s use x = true ->
  value_of_definition s index' valuation x = None.
Proof.
  intros [_ [Hnew [[Hrecorded _] _]]] Huse Hknown.
  unfold value_of_definition.
  destruct (lookup x (Join_env.definitions_of_existentials s))
    as [[simples kind] | ] eqn:Hdefinition; [ | reflexivity ].
  exfalso.
  assert (Hx : is_existential s x).
  { unfold is_existential. apply is_bound_lookup. exists kind.
    exact (Hrecorded _ _ _ Hdefinition). }
  destruct (Hnew x Hx) as [Hsource Huses].
  unfold known_to in Hknown. apply orb_true_iff in Hknown.
  destruct Hknown as [Hk | Hk];
    [ exact (Huses _ _ Huse Hk) | exact (Hsource Hk) ].
Qed.

Lemma target_valuation_known (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) (x : variable) :
  state_valid s ->
  Join_env.joined_env s index = Some use ->
  known_to s use x = true ->
  lookup x (target_valuation s index use valuation) = lookup x valuation.
Proof.
  intros Hvalid Huse Hknown. rewrite lookup_target_valuation.
  rewrite (known_no_definition s index index use valuation x Hvalid Huse
             Hknown).
  rewrite Hknown. reflexivity.
Qed.

(* The valuation of the target extends that of the input. *)
Lemma target_valuation_extends_at (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) :
  state_valid s ->
  Join_env.joined_env s index = Some use ->
  extends_input Join_env.joined_env Join_env.definitions_of_existentials s
    index valuation (target_valuation s index use valuation)
  ∧ agrees_on_known (Join_env.source_env s) valuation
      (target_valuation s index use valuation).
Proof.
  intros Hvalid Huse. split.
  - exists use. split; [ exact Huse | ]. split.
    + intros x Hknown. apply (target_valuation_known s index use valuation x
                                Hvalid Huse).
      unfold known_to. unfold knows in Hknown. rewrite Hknown. reflexivity.
    + intros x simples kind simple v Hdefinition Hfind Hv.
      rewrite lookup_target_valuation. unfold value_of_definition.
      rewrite Hdefinition, Hfind, Hv. reflexivity.
  - intros x Hknown. apply (target_valuation_known s index use valuation x
                              Hvalid Huse).
    unfold known_to. unfold knows in Hknown. rewrite Hknown.
    apply orb_true_r.
Qed.

(* A simple in scope of an input has the same value in both. *)
Lemma target_valuation_simple (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) (simple : simple) :
  state_valid s ->
  Join_env.joined_env s index = Some use ->
  simple_in_scope use simple ->
  eval_simple (target_valuation s index use valuation) simple
  = eval_simple valuation simple.
Proof.
  intros Hvalid Huse Hscope. destruct simple as [x | sym | c];
    [ | reflexivity | reflexivity ].
  cbn. apply (target_valuation_known s index use valuation x Hvalid Huse).
  unfold known_to. rewrite (Hscope x (or_introl eq_refl)). reflexivity.
Qed.

(* A type in scope of an input means the same in both. *)
Lemma target_valuation_type (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (statics : static_env) (valuation : env) (v : value)
    (ty : flambda_type) :
  state_valid s ->
  Join_env.joined_env s index = Some use ->
  type_in_scope use ty ->
  type_satisfied statics valuation v ty ->
  type_satisfied statics (target_valuation s index use valuation) v ty.
Proof.
  intros Hvalid Huse Hscope Hty.
  apply (type_satisfied_coincide statics valuation
           (target_valuation s index use valuation) v ty); [ | exact Hty ].
  intros x Hx. symmetry.
  apply (target_valuation_known s index use valuation x Hvalid Huse).
  unfold known_to. rewrite (Hscope x Hx). reflexivity.
Qed.

(* ---------------------------------------------------------------------- *)
(* Growth                                                                  *)
(* ---------------------------------------------------------------------- *)

Abbreviation grows :=
  (state_grows Join_env.joined_env Join_env.definitions_of_existentials
     kind_in_target_env).

Abbreviation exists_in_target := (exists_in_target_env kind_in_target_env).

(* A later state of one join: both are valid, the second has grown,
   and its source is the same. *)
Definition later (s s' : Join_env.t) : Prop :=
  state_valid s
  ∧ state_valid s'
  ∧ grows s s'
  ∧ Join_env.source_env s' = Join_env.source_env s.

Lemma later_refl (s : Join_env.t) : state_valid s -> later s s.
Proof.
  intros Hvalid. split; [ exact Hvalid | ]. split; [ exact Hvalid | ].
  split; [ apply grows_refl | reflexivity ].
Qed.

Lemma later_trans (s1 s2 s3 : Join_env.t) :
  later s1 s2 -> later s2 s3 -> later s1 s3.
Proof.
  intros [Hvalid [_ [Hgrows Hsource]]] [_ [Hvalid' [Hgrows' Hsource']]].
  split; [ exact Hvalid | ]. split; [ exact Hvalid' | ].
  split; [ exact (grows_trans _ _ _ Hgrows Hgrows') | ].
  rewrite Hsource'. exact Hsource.
Qed.

Lemma later_exists (s s' : Join_env.t) (x : variable) :
  later s s' -> exists_in_target s x = true -> exists_in_target s' x = true.
Proof. intros [_ [_ [Hgrows _]]]. exact (grows_exists _ _ _ Hgrows). Qed.

Lemma later_env (s s' : Join_env.t) (index : Index.t) :
  later s s' -> Join_env.joined_env s' index = Join_env.joined_env s index.
Proof. intros [_ [_ [[Henvs _] _]]]. exact (Henvs index). Qed.

Lemma later_reach (s s' : Join_env.t) (x : variable) :
  later s s' -> in_reach s x -> in_reach s' x.
Proof.
  intros Hlater [Hexists | [index [use [Huse Hknown]]]].
  - left. exact (later_exists _ _ _ Hlater Hexists).
  - right. exists index, use. rewrite (later_env _ _ _ Hlater).
    split; [ exact Huse | exact Hknown ].
Qed.

Lemma existential_in_reach (s : Join_env.t) (e : variable) :
  is_existential s e -> exists_in_target s e = true.
Proof.
  intros Hexistential. rewrite kind_in_target_env_exists.
  unfold Bindings_in_target_env.exists_in_target_env.
  unfold is_existential in Hexistential. rewrite Hexistential.
  rewrite orb_true_r. reflexivity.
Qed.

(* A variable in reach has a definition later only if it has it
   now. *)
Lemma later_definition (s s' : Join_env.t) (x : variable) :
  later s s' ->
  in_reach s x ->
  lookup x (Join_env.definitions_of_existentials s')
  = lookup x (Join_env.definitions_of_existentials s).
Proof.
  intros [Hvalid [Hvalid' [[Henvs [Hdefinitions Hkinds]] Hsource]]] Hreach.
  destruct (lookup x (Join_env.definitions_of_existentials s))
    as [definition | ] eqn:Hdefinition.
  - exact (Hdefinitions _ _ Hdefinition).
  - destruct (lookup x (Join_env.definitions_of_existentials s'))
      as [[simples kind] | ] eqn:Hdefinition'; [ | reflexivity ].
    exfalso.
    destruct Hvalid' as [_ [Hnew' [[Hrecorded' _] _]]].
    assert (Hx' : is_existential s' x).
    { unfold is_existential. apply is_bound_lookup. exists kind.
      exact (Hrecorded' _ _ _ Hdefinition'). }
    destruct (Hnew' x Hx') as [Hsource' Huses'].
    destruct Hreach as [Hexists | [index [use [Huse Hknown]]]].
    + rewrite kind_in_target_env_exists in Hexists.
      unfold Bindings_in_target_env.exists_in_target_env in Hexists.
      apply orb_true_iff in Hexists. destruct Hexists as [Hexists | Himported].
      * apply orb_true_iff in Hexists. destruct Hexists as [Hin | Hexistential].
        -- apply Hsource'. rewrite Hsource. exact Hin.
        -- destruct Hvalid as [_ [_ [[_ [_ Hdefined]] _]]].
           pose proof (Hdefined x Hexistential) as Hbound.
           unfold is_bound in Hbound. rewrite Hdefinition in Hbound.
           discriminate.
      * destruct Hvalid as [_ [_ [_ [_ Himports]]]].
        destruct (Himports x Himported) as [_ [index [use [Huse Hknown]]]].
        apply (Huses' index use); [ | exact Hknown ].
        rewrite Henvs. exact Huse.
    + apply (Huses' index use); [ | exact Hknown ].
      rewrite Henvs. exact Huse.
Qed.

Lemma later_target_valuation (s s' : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) (x : variable) :
  later s s' ->
  in_reach s x ->
  lookup x (target_valuation s' index use valuation)
  = lookup x (target_valuation s index use valuation).
Proof.
  intros Hlater Hreach.
  rewrite !lookup_target_valuation. unfold value_of_definition.
  rewrite (later_definition s s' x Hlater Hreach).
  unfold known_to. rewrite (proj2 (proj2 (proj2 Hlater))). reflexivity.
Qed.

(* The meaning of a type whose variables are in reach is the same
   later. *)
Lemma later_type (s s' : Join_env.t) (index : Index.t) (use : Typing_env.t)
    (statics : static_env) (valuation : env) (v : value) (ty : flambda_type) :
  later s s' ->
  (∀ y : variable, y ∈ free_variables ty -> in_reach s y) ->
  type_satisfied statics (target_valuation s index use valuation) v ty ->
  type_satisfied statics (target_valuation s' index use valuation) v ty.
Proof.
  intros Hlater Hreach Hty.
  apply (type_satisfied_coincide statics
           (target_valuation s index use valuation)
           (target_valuation s' index use valuation) v ty); [ | exact Hty ].
  intros y Hy. symmetry.
  exact (later_target_valuation s s' index use valuation y Hlater
           (Hreach y Hy)).
Qed.

Lemma later_tags_at (s s' : Join_env.t) (use : Typing_env.t)
    (statics : static_env) (valuation : env) :
  later s s' ->
  tags_at s' use statics valuation ->
  tags_at s use statics valuation.
Proof.
  intros Hlater Htags. unfold tags_at, known_to in *.
  rewrite (proj2 (proj2 (proj2 Hlater))) in Htags. exact Htags.
Qed.

Lemma later_equation_holds (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  later s s' -> equation_holds s x ty -> equation_holds s' x ty.
Proof.
  intros Hlater [Hexists [Hreach Hholds]].
  split; [ exact (later_exists _ _ _ Hlater Hexists) | ].
  split; [ intros y Hy; exact (later_reach _ _ _ Hlater (Hreach y Hy)) | ].
  intros index use statics valuation v Huse Hsatisfied Htags Hv.
  rewrite (later_env _ _ _ Hlater) in Huse.
  rewrite (later_target_valuation s s' index use valuation x Hlater
             (or_introl Hexists)) in Hv.
  exact (later_type s s' index use statics valuation v ty Hlater Hreach
           (Hholds _ _ _ _ _ Huse Hsatisfied
              (later_tags_at _ _ _ _ _ Hlater Htags) Hv)).
Qed.

Lemma later_types_hold (s s' : Join_env.t) (x : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  later s s' -> types_hold s x types -> types_hold s' x types.
Proof.
  intros Hlater [Hexists [Hfound Htrue]].
  split; [ exact (later_exists _ _ _ Hlater Hexists) | ]. split.
  - intros index t Ht. rewrite (later_env _ _ _ Hlater).
    exact (Hfound index t Ht).
  - intros index use statics valuation v Huse Hsatisfied Hv.
    rewrite (later_env _ _ _ Hlater) in Huse.
    rewrite (later_target_valuation s s' index use valuation x Hlater
               (or_introl Hexists)) in Hv.
    exact (Htrue _ _ _ _ _ Huse Hsatisfied Hv).
Qed.

(* ---------------------------------------------------------------------- *)
(* The joined environments, with their levels, do not change               *)
(* ---------------------------------------------------------------------- *)

Abbreviation same_envs s s' :=
  (Join_env.joined_envs s' = Join_env.joined_envs s).

Lemma same_envs_trans (s1 s2 s3 : Join_env.t) :
  same_envs s1 s2 -> same_envs s2 s3 -> same_envs s1 s3.
Proof. intros H12 H23. rewrite H23. exact H12. Qed.

Lemma same_readings_same_envs (s s' : Join_env.t) :
  same_readings s s' -> same_envs s s'.
Proof. intros [H _]. exact H. Qed.

Lemma existential_same_envs (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  same_envs s s'.
Proof.
  unfold Join_env.existential_for_these_canonicals. intros Hmake.
  destruct (Bindings_in_target_env.existential_for_these_simples
              (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
    as [x0 bindings0].
  injection Hmake as _ <-. reflexivity.
Qed.

Lemma add_alias_same_envs (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  same_envs s (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [x s1] eqn:Hmake.
  exact (existential_same_envs _ _ _ _ _ _ Hmake).
Qed.

Definition import_same_envs
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    import_var s x = Some s' -> same_envs s s'.

Lemma import_vars_same_envs
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_same_envs import_var ->
  ∀ s s' : Join_env.t,
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    same_envs s s'.
Proof.
  intros Himport.
  induction vars as [ | var vars IH]; intros s s' Hfold.
  - injection Hfold as <-. reflexivity.
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + exact (same_envs_trans _ _ _ (Himport _ _ _ Hvar) (IH _ _ Hfold)).
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma add_equation_same_envs (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' -> same_envs s s'.
Proof.
  intros Hadd.
  exact (same_readings_same_envs _ _ (add_equation_same_readings _ _ _ _ Hadd)).
Qed.

Lemma add_equals_same_envs (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  import_same_envs import_var ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  same_envs s s'.
Proof.
  intros Himport Hadd.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ]; [ | | | discriminate ].
  - exact (add_equation_same_envs _ _ _ _ Hadd).
  - destruct (import_var s imported) as [s1 | ] eqn:Hvar; [ | discriminate ].
    exact (same_envs_trans _ _ _ (Himport _ _ _ Hvar)
             (add_equation_same_envs _ _ _ _ Hadd)).
  - injection Hadd as <-. apply add_alias_same_envs.
Qed.

Lemma concrete_same_envs (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  import_same_envs import_var ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  same_envs s s'.
Proof.
  intros Himport Hconcrete.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    exact (same_envs_trans _ _ _ (add_equation_same_envs _ _ _ _ Hadd)
             (import_vars_same_envs _ _ Himport _ _ Hconcrete)).
  - injection Hconcrete as <-. reflexivity.
Qed.

Lemma import_body_same_envs (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  import_same_envs import_var ->
  import_same_envs (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Himport s s' x Hbody.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x);
    [ injection Hbody as <-; reflexivity | ].
  destruct (among x (Join_env.imported_vars s));
    [ injection Hbody as <-; reflexivity | ].
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ]; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  assert (Hfirst : same_envs s (with_import s x kind)) by reflexivity.
  apply (same_envs_trans _ _ _ Hfirst).
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ];
    [ | discriminate ].
  destruct (existsb _ _).
  - destruct
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         _ _) as [[canonical_simples | ] | ]; try discriminate.
    exact (add_equals_same_envs _ _ _ _ _ _ _ Himport Hbody).
  - exact (concrete_same_envs _ _ _ _ _ _ Himport Hbody).
Qed.

Lemma import_var_same_envs (p : Join_env.parameters) (fuel : nat) :
  import_same_envs (Join_env.import_var_transitive p fuel).
Proof.
  induction fuel as [ | fuel IH].
  - intros s s' x H. discriminate.
  - exact (import_body_same_envs p _ IH).
Qed.

Lemma import_or_delay_same_envs (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  same_envs s s'.
Proof.
  intros Himport.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in Himport.
  destruct (Index.choose types) as [[index0 any_ty] | ]; [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
       (Join_env.joined_envs s) types)
    as [[canonicals | ] | ]; [ | | discriminate ].
  - exact (add_equals_same_envs _ _ _ _ _ _ _ (import_var_same_envs p fuel)
             Himport).
  - exact (concrete_same_envs _ _ _ _ _ _ (import_var_same_envs p fuel)
             Himport).
Qed.

Lemma later_existential (s s' : Join_env.t) (e : variable) :
  later s s' -> is_existential s e -> is_existential s' e.
Proof.
  intros [Hvalid [Hvalid' [[_ [Hdefinitions _]] _]]] Hexistential.
  destruct Hvalid as [_ [_ [[_ [_ Hdefined]] _]]].
  pose proof (Hdefined e Hexistential) as Hbound.
  apply is_bound_lookup in Hbound. destruct Hbound as [[simples kind] Hd].
  destruct Hvalid' as [_ [_ [[Hrecorded' _] _]]].
  unfold is_existential. apply is_bound_lookup. exists kind.
  exact (Hrecorded' _ _ _ (Hdefinitions _ _ Hd)).
Qed.

Lemma later_alias_holds (s s' : Join_env.t) (e x : variable) :
  later s s' -> same_envs s s' -> alias_holds s e x -> alias_holds s' e x.
Proof.
  intros Hlater Henvs [Hexists [Hexistential [Hsame [Hlocated Hjoined]]]].
  split; [ exact (later_exists _ _ _ Hlater Hexists) | ].
  split; [ exact (later_existential _ _ _ Hlater Hexistential) | ].
  split; [ | split ].
  - intros index use statics valuation v Huse Hsatisfied Hv.
    rewrite (later_env _ _ _ Hlater) in Huse.
    rewrite (later_target_valuation s s' index use valuation x Hlater
               (or_introl Hexists)) in Hv.
    rewrite (later_target_valuation s s' index use valuation e Hlater
               (or_introl (existential_in_reach _ _ Hexistential))).
    exact (Hsame _ _ _ _ _ Huse Hsatisfied Hv).
  - intros simples kind index simple Hdefinition Hfind.
    pose proof Hlater as [Hvalid [_ [[_ [Hdefinitions _]] _]]].
    destruct Hvalid as [_ [_ [[_ [_ Hdefined]] _]]].
    pose proof (Hdefined e Hexistential) as Hbound.
    apply is_bound_lookup in Hbound.
    destruct Hbound as [[simples0 kind0] Hd].
    pose proof (Hdefinitions _ _ Hd) as Hd'. rewrite Hdefinition in Hd'.
    injection Hd' as <- <-.
    destruct (Hlocated _ _ _ _ Hd Hfind) as [use [Huse Hknows]].
    exists use.
    split; [ rewrite (later_env _ _ _ Hlater); exact Huse | exact Hknows ].
  - intros Hknown index use equations Hlookup.
    rewrite (proj2 (proj2 (proj2 Hlater))) in Hknown.
    rewrite Henvs in Hlookup.
    exact (Hjoined Hknown index use equations Hlookup).
Qed.

Lemma later_vector_in_envs (s s' : Join_env.t)
    (simples : Simples_in_joined_envs.t) :
  later s s' -> vector_in_envs s simples -> vector_in_envs s' simples.
Proof.
  intros Hlater Hvector index simple Hfind.
  rewrite (later_env _ _ _ Hlater). exact (Hvector index simple Hfind).
Qed.

(* What the state means is kept later, where the four maps are the
   same. *)
Lemma later_holds (s s' : Join_env.t) :
  later s s' ->
  same_envs s s' ->
  Join_env.types_in_target_env s' = Join_env.types_in_target_env s ->
  Join_env.types_in_joined_envs s' = Join_env.types_in_joined_envs s ->
  Join_env.aliases_of_existentials s' = Join_env.aliases_of_existentials s ->
  Join_env.definitions_of_existentials s'
  = Join_env.definitions_of_existentials s ->
  state_holds s ->
  state_holds s'.
Proof.
  intros Hlater Henvs Htarget Hjoined Haliases Hdefinitions
    [Hequations [Htypes [Halias Hvectors]]].
  split; [ | split; [ | split ] ].
  - intros x ty Hx. rewrite Htarget in Hx.
    exact (later_equation_holds _ _ _ _ Hlater (Hequations x ty Hx)).
  - intros x types Hx. rewrite Hjoined in Hx.
    exact (later_types_hold _ _ _ _ Hlater (Htypes x types Hx)).
  - intros e x Hx. rewrite Haliases in Hx.
    exact (later_alias_holds _ _ _ _ Hlater Henvs (Halias e x Hx)).
  - intros x simples kind Hx. rewrite Hdefinitions in Hx.
    exact (later_vector_in_envs _ _ _ Hlater (Hvectors x simples kind Hx)).
Qed.

(* ---------------------------------------------------------------------- *)
(* The steps                                                               *)
(* ---------------------------------------------------------------------- *)

Abbreviation envs_invariant := (joined_envs_are typing_env_invariant).

Abbreviation envs_kinded := (joined_envs_are typing_env_kinded).

Lemma later_envs_invariant (s s' : Join_env.t) :
  later s s' -> envs_invariant s -> envs_invariant s'.
Proof.
  intros Hlater H index use Huse. rewrite (later_env _ _ _ Hlater) in Huse.
  exact (H _ _ Huse).
Qed.

Lemma later_envs_kinded (s s' : Join_env.t) :
  later s s' -> envs_kinded s -> envs_kinded s'.
Proof.
  intros Hlater H index use Huse. rewrite (later_env _ _ _ Hlater) in Huse.
  exact (H _ _ Huse).
Qed.

Lemma same_readings_later (s s' : Join_env.t) :
  same_readings s s' -> state_valid s -> later s s'.
Proof.
  intros Hsame Hvalid. split; [ exact Hvalid | ].
  split; [ exact (same_readings_valid _ _ Hsame Hvalid) | ].
  split; [ exact (same_readings_grows _ _ Hsame) | ].
  destruct Hsame as [_ [Hbindings _]]. unfold Join_env.source_env.
  rewrite Hbindings. reflexivity.
Qed.

Lemma same_readings_target_valuation (s s' : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) :
  same_readings s s' ->
  target_valuation s' index use valuation
  = target_valuation s index use valuation.
Proof.
  intros [_ [Hbindings [Hdefinitions _]]].
  unfold target_valuation, value_of_definition, known_to, Join_env.source_env.
  rewrite Hbindings, Hdefinitions. reflexivity.
Qed.

(* What the state means is kept where each new entry of the three maps
   is true. *)
Lemma holds_with (s s' : Join_env.t) :
  later s s' ->
  same_envs s s' ->
  state_holds s ->
  (∀ (x : variable) (ty : flambda_type),
     lookup x (Join_env.types_in_target_env s') = Some ty ->
     lookup x (Join_env.types_in_target_env s) = Some ty
     ∨ equation_holds s' x ty) ->
  (∀ (x : variable) (types : Keyed.finite_map Index.t flambda_type),
     lookup x (Join_env.types_in_joined_envs s') = Some types ->
     lookup x (Join_env.types_in_joined_envs s) = Some types
     ∨ types_hold s' x types) ->
  (∀ e x : variable,
     x
     ∈ Aliases_of_existentials.aliases_of_existential_var
         (Join_env.aliases_of_existentials s') e ->
     x
     ∈ Aliases_of_existentials.aliases_of_existential_var
         (Join_env.aliases_of_existentials s) e
     ∨ alias_holds s' e x) ->
  (∀ (x : variable) (simples : Simples_in_joined_envs.t)
     (kind : flambda_kind),
     lookup x (Join_env.definitions_of_existentials s') = Some (simples, kind)
     ->
     lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind)
     ∨ vector_in_envs s' simples) ->
  state_holds s'.
Proof.
  intros Hlater Henvs [Hequations [Htypes [Halias Hvectors]]] Htarget Hjoined
    Haliases Hdefinitions.
  split; [ | split; [ | split ] ].
  - intros x ty Hx. destruct (Htarget x ty Hx) as [Hold | Hnew];
      [ | exact Hnew ].
    exact (later_equation_holds _ _ _ _ Hlater (Hequations x ty Hold)).
  - intros x types Hx. destruct (Hjoined x types Hx) as [Hold | Hnew];
      [ | exact Hnew ].
    exact (later_types_hold _ _ _ _ Hlater (Htypes x types Hold)).
  - intros e x Hx. destruct (Haliases e x Hx) as [Hold | Hnew];
      [ | exact Hnew ].
    exact (later_alias_holds _ _ _ _ Hlater Henvs (Halias e x Hold)).
  - intros x simples kind Hx.
    destruct (Hdefinitions x simples kind Hx) as [Hold | Hnew];
      [ | exact Hnew ].
    exact (later_vector_in_envs _ _ _ Hlater (Hvectors x simples kind Hold)).
Qed.

Lemma add_equation_holds (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  state_valid s ->
  state_holds s ->
  equation_holds s x ty ->
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  later s s' ∧ state_holds s'.
Proof.
  intros Hvalid Hholds Hequation Hadd.
  pose proof (same_readings_later _ _
                (add_equation_same_readings _ _ _ _ Hadd) Hvalid) as Hlater.
  split; [ exact Hlater | ].
  unfold Join_env.add_equation_in_target_env in Hadd.
  destruct (is_bound x (Join_env.types_in_target_env s)); [ discriminate | ].
  injection Hadd as <-.
  apply (holds_with s _ Hlater eq_refl Hholds);
    cbn [Join_env.types_in_target_env Join_env.types_in_joined_envs
           Join_env.aliases_of_existentials
           Join_env.definitions_of_existentials].
  - intros x' ty' Hx'. rewrite lookup_cons in Hx'.
    destruct (Pos.eqb x' x) eqn:Heq; [ | left; exact Hx' ].
    injection Hx' as <-. apply Pos.eqb_eq in Heq. subst x'. right.
    exact (later_equation_holds _ _ _ _ Hlater Hequation).
  - intros x' types Hx'. left. exact Hx'.
  - intros e x' Hx'. left. exact Hx'.
  - intros x' simples kind Hx'. left. exact Hx'.
Qed.

Lemma add_equations_holds (s : Join_env.t) (x : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  state_valid s ->
  state_holds s ->
  types_hold s x types ->
  later s (Join_env.add_equations_in_joined_envs s x types)
  ∧ state_holds (Join_env.add_equations_in_joined_envs s x types).
Proof.
  intros Hvalid Hholds Htypes.
  pose proof (same_readings_later _ _
                (add_equations_same_readings s x types) Hvalid) as Hlater.
  split; [ exact Hlater | ].
  apply (holds_with s _ Hlater eq_refl Hholds);
    cbn [Join_env.add_equations_in_joined_envs Join_env.types_in_target_env
           Join_env.types_in_joined_envs Join_env.aliases_of_existentials
           Join_env.definitions_of_existentials].
  - intros x' ty' Hx'. left. exact Hx'.
  - intros x' types' Hx'. rewrite lookup_cons in Hx'.
    destruct (Pos.eqb x' x) eqn:Heq; [ | left; exact Hx' ].
    injection Hx' as <-. apply Pos.eqb_eq in Heq. subst x'. right.
    exact (later_types_hold _ _ _ _ Hlater Htypes).
  - intros e x' Hx'. left. exact Hx'.
  - intros x' simples kind Hx'. left. exact Hx'.
Qed.

Lemma existential_same_maps (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  Join_env.source_env s' = Join_env.source_env s
  ∧ Join_env.types_in_target_env s' = Join_env.types_in_target_env s
  ∧ Join_env.types_in_joined_envs s' = Join_env.types_in_joined_envs s
  ∧ Join_env.aliases_of_existentials s' = Join_env.aliases_of_existentials s.
Proof.
  unfold Join_env.existential_for_these_canonicals,
    Bindings_in_target_env.existential_for_these_simples.
  intros Hmake.
  destruct (Bindings_in_target_env.has_existential_for_these_simples
              (Join_env.bindings s) canonicals);
    injection Hmake as _ <-; repeat split; reflexivity.
Qed.

Lemma existential_definitions (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  ∀ (y : variable) (simples : Simples_in_joined_envs.t)
    (kind' : flambda_kind),
    lookup y (Join_env.definitions_of_existentials s') = Some (simples, kind')
    ->
    lookup y (Join_env.definitions_of_existentials s) = Some (simples, kind')
    ∨ simples = canonicals.
Proof.
  unfold Join_env.existential_for_these_canonicals. intros Hmake.
  destruct (Bindings_in_target_env.existential_for_these_simples
              (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
    as [x0 bindings0].
  injection Hmake as _ <-.
  intros y simples kind' Hy. cbn [Join_env.definitions_of_existentials] in Hy.
  destruct (is_bound x0 (Join_env.definitions_of_existentials s));
    [ left; exact Hy | ].
  rewrite lookup_cons in Hy.
  destruct (Pos.eqb y x0); [ | left; exact Hy ].
  injection Hy as <- _. right. reflexivity.
Qed.

Lemma existential_holds (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  fresh_avoids p ->
  state_valid s ->
  state_holds s ->
  vector_in_envs s canonicals ->
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  later s s' ∧ state_holds s'.
Proof.
  intros Hfresh Hvalid Hholds Hvector Hmake.
  destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
    as [Hvalid' [Hgrows _]].
  destruct (existential_same_maps _ _ _ _ _ _ Hmake)
    as [Hsource [Htarget [Hjoined Haliases]]].
  assert (Hlater : later s s').
  { split; [ exact Hvalid | ]. split; [ exact Hvalid' | ].
    split; [ exact Hgrows | exact Hsource ]. }
  split; [ exact Hlater | ].
  apply (holds_with s s' Hlater (existential_same_envs _ _ _ _ _ _ Hmake)
           Hholds).
  - intros y ty Hy. left. rewrite Htarget in Hy. exact Hy.
  - intros y types Hy. left. rewrite Hjoined in Hy. exact Hy.
  - intros e y Hy. left. rewrite Haliases in Hy. exact Hy.
  - intros y simples kind' Hy.
    destruct (existential_definitions _ _ _ _ _ _ Hmake _ _ _ Hy)
      as [Hold | ->]; [ left; exact Hold | right ].
    exact (later_vector_in_envs _ _ _ Hlater Hvector).
Qed.

(* A vector of [Simple]s stands for a variable of the target: at an
   input where the variable has a value, of the kind given, the vector
   has the canonical element of a [Simple] that has that value. *)
Definition vector_holds (s : Join_env.t) (x : variable) (kind : flambda_kind)
    (simples : Simples_in_joined_envs.t) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
    (valuation : env) (v : value),
    Join_env.joined_env s index = Some use ->
    typing_env_satisfied_strictly statics valuation use ->
    lookup x (target_valuation s index use valuation) = Some v ->
    kind_of_value v = kind
    ∧ ∃ simple : simple,
        simple_in_scope use simple
        ∧ Simples_in_joined_envs.find index simples
          = Some (Typing_env.get_canonical_simple_ignoring_name_mode use simple)
        ∧ eval_simple valuation simple = Some v.

Lemma later_vector_holds (s s' : Join_env.t) (x : variable)
    (kind : flambda_kind) (simples : Simples_in_joined_envs.t) :
  later s s' ->
  exists_in_target s x = true ->
  vector_holds s x kind simples ->
  vector_holds s' x kind simples.
Proof.
  intros Hlater Hexists Hvector index use statics valuation v Huse Hsatisfied
    Hv.
  rewrite (later_env _ _ _ Hlater) in Huse.
  rewrite (later_target_valuation s s' index use valuation x Hlater
             (or_introl Hexists)) in Hv.
  exact (Hvector _ _ _ _ _ Huse Hsatisfied Hv).
Qed.

Lemma aliases_add_cases (aliases : Aliases_of_existentials.t)
    (demoted e e' x : variable) :
  x
  ∈ Aliases_of_existentials.aliases_of_existential_var
      (Aliases_of_existentials.add aliases demoted e) e' ->
  x ∈ Aliases_of_existentials.aliases_of_existential_var aliases e'
  ∨ (e' = e ∧ x = demoted).
Proof.
  unfold Aliases_of_existentials.add.
  destruct (among demoted
              (Aliases_of_existentials.aliases_of_existential_var aliases e));
    [ intros H; left; exact H | ].
  unfold Aliases_of_existentials.aliases_of_existential_var at 1.
  rewrite lookup_cons. destruct (Pos.eqb e' e) eqn:Heq.
  - apply Pos.eqb_eq in Heq. subst e'. intros [Hx | Hx].
    + right. split; [ reflexivity | symmetry; exact Hx ].
    + left. exact Hx.
  - intros H. left. exact H.
Qed.

(* Every input at which a vector has a [Simple] knows a variable. *)
Definition vector_known (s : Join_env.t) (x : variable)
    (simples : Simples_in_joined_envs.t) : Prop :=
  ∀ (index : Index.t) (simple : simple),
    Simples_in_joined_envs.find index simples = Some simple ->
    ∃ use : Typing_env.t,
      Join_env.joined_env s index = Some use ∧ knows use x.

Lemma add_alias_holds (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  state_valid s ->
  state_holds s ->
  exists_in_target s var = true ->
  vector_holds s var kind canonicals ->
  vector_in_envs s canonicals ->
  vector_known s var canonicals ->
  (knows (Join_env.source_env s) var -> joined_everywhere s var) ->
  later s (Join_env.add_alias_of_existential p s kind var canonicals)
  ∧ state_holds (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  intros Hfresh Hvalid Hholds Hvar Hvector Henvs Hknown Hlevels.
  unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [e s1] eqn:Hmake.
  destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
    as [_ [_ [Hexistential [simples' [kind' [Hdefinition Hequal]]]]]].
  destruct (existential_holds _ _ _ _ _ _ Hfresh Hvalid Hholds Henvs Hmake)
    as [Hlater1 Hholds1].
  match goal with
  | |- later s ?s' ∧ _ =>
    assert (Hsame : same_readings s1 s') by (repeat split);
    set (s2 := s') in *
  end.
  pose proof (same_readings_later _ _ Hsame (proj1 (proj2 Hlater1)))
    as Hlater2.
  split; [ exact (later_trans _ _ _ Hlater1 Hlater2) | ].
  apply (holds_with s1 s2 Hlater2 (proj1 Hsame) Hholds1).
  - intros x ty Hx. left. exact Hx.
  - intros x types Hx. left. exact Hx.
  - intros e' x Hx.
    change (Join_env.aliases_of_existentials s2)
      with (Aliases_of_existentials.add (Join_env.aliases_of_existentials s1)
              var e) in Hx.
    destruct (aliases_add_cases _ _ _ _ _ Hx) as [Hold | [-> ->]];
      [ left; exact Hold | right ].
    pose proof (later_trans _ _ _ Hlater1 Hlater2) as Hlater.
    split; [ exact (later_exists _ _ _ Hlater Hvar) | ].
    split; [ exact (later_existential _ _ _ Hlater2 Hexistential) | ].
    split; [ | split ].
    + intros index use statics valuation v Huse Hsatisfied Hv.
      rewrite (later_env _ _ _ Hlater) in Huse.
      rewrite (later_target_valuation s s2 index use valuation var Hlater
                 (or_introl Hvar)) in Hv.
      destruct (Hvector _ _ _ _ _ Huse Hsatisfied Hv)
        as [_ [simple [Hscope [Hfind Hvalue]]]].
      rewrite (same_readings_target_valuation _ _ _ _ _ Hsame).
      rewrite lookup_target_valuation. unfold value_of_definition.
      rewrite Hdefinition.
      rewrite <- (simples_equal_find _ _ Hequal index), Hfind.
      rewrite (canonical_ignoring_name_mode_satisfied _ _ _ _ simple
                 Hsatisfied).
      rewrite Hvalue. reflexivity.
    + intros simples0 kind0 index simple0 Hdefinition0 Hfind0.
      rewrite (proj1 (proj2 (proj2 Hsame))) in Hdefinition0.
      rewrite Hdefinition in Hdefinition0. injection Hdefinition0 as <- <-.
      rewrite <- (simples_equal_find _ _ Hequal index) in Hfind0.
      destruct (Hknown index simple0 Hfind0) as [use [Huse Hknows]].
      exists use.
      split; [ rewrite (later_env _ _ _ Hlater); exact Huse | exact Hknows ].
    + intros Hsource index use equations Hlookup.
      rewrite (proj2 (proj2 (proj2 Hlater))) in Hsource.
      rewrite (proj1 Hsame), (existential_same_envs _ _ _ _ _ _ Hmake)
        in Hlookup.
      exact (Hlevels Hsource index use equations Hlookup).
  - intros x simples k Hx. left. exact Hx.
Qed.

Definition import_holds
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    state_valid s ->
    envs_invariant s ->
    envs_kinded s ->
    state_holds s ->
    import_var s x = Some s' ->
    later s s' ∧ exists_in_target s' x = true ∧ state_holds s'.

Lemma import_vars_holds
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_holds import_var ->
  ∀ s s' : Join_env.t,
    state_valid s ->
    envs_invariant s ->
    envs_kinded s ->
    state_holds s ->
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    later s s' ∧ state_holds s'.
Proof.
  intros Himport.
  induction vars as [ | var vars IH];
    intros s s' Hvalid Hinvariant Hkinded Hholds Hfold.
  - injection Hfold as <-.
    split; [ exact (later_refl s Hvalid) | exact Hholds ].
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + destruct (Himport _ _ _ Hvalid Hinvariant Hkinded Hholds Hvar)
        as [Hlater1 [_ Hholds1]].
      destruct (IH _ _ (proj1 (proj2 Hlater1))
                  (later_envs_invariant _ _ Hlater1 Hinvariant)
                  (later_envs_kinded _ _ Hlater1 Hkinded) Hholds1 Hfold)
        as [Hlater' Hholds'].
      split; [ exact (later_trans _ _ _ Hlater1 Hlater') | exact Hholds' ].
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma free_variables_alias_type_of (kind : flambda_kind) (simple : simple) :
  free_variables (alias_type_of kind simple) = free_variables_of_simple simple.
Proof.
  destruct kind as [ | k | | ]; [ | destruct k | | ]; reflexivity.
Qed.

Lemma source_exists (s : Join_env.t) (x : variable) :
  Source_env.exists_in_source_env (Join_env.source_env s) x = true ->
  exists_in_target s x = true.
Proof.
  rewrite kind_in_target_env_exists.
  unfold Bindings_in_target_env.exists_in_target_env, Join_env.source_env.
  intros ->. reflexivity.
Qed.

Lemma later_stands_in_for (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (joined : simple) :
  later s s' ->
  stands_in_for s canonicals joined ->
  stands_in_for s' canonicals joined.
Proof.
  intros Hlater Hstands index canonical Hfind.
  destruct (Hstands _ _ Hfind) as [Heq | [env [Henv Heq]]];
    [ left; exact Heq | right ].
  exists env. rewrite (later_env _ _ _ Hlater).
  split; [ exact Henv | exact Heq ].
Qed.

(* The alias equation of a variable to what stands for its vector. *)
Lemma alias_equation_holds (s : Join_env.t) (var : variable)
    (kind : flambda_kind) (simples : Simples_in_joined_envs.t)
    (joined : simple) :
  state_valid s ->
  envs_invariant s ->
  exists_in_target s var = true ->
  simple_in_target kind_in_target_env s joined ->
  vector_holds s var kind simples ->
  stands_in_for s simples joined ->
  equation_holds s var (alias_type_of kind joined).
Proof.
  intros Hvalid Hinvariant Hvar Hjoined Hvector Hstands.
  split; [ exact Hvar | ]. split.
  - intros y Hy. rewrite free_variables_alias_type_of in Hy. left.
    exact (Hjoined y Hy).
  - intros index use statics valuation v Huse Hsatisfied _ Hv.
    destruct (Hvector _ _ _ _ _ Huse Hsatisfied Hv)
      as [Hkind [simple [Hscope [Hfind Hvalue]]]].
    apply alias_satisfied. split; [ | exact Hkind ].
    destruct (target_valuation_extends_at s index use valuation Hvalid Huse)
      as [[use' [Huse' [Hagrees _]]] _].
    rewrite Huse in Huse'. injection Huse' as <-.
    exact (stands_upper s simples joined index simple use statics valuation _
             v Hstands (Hinvariant _ _ Huse) Huse Hscope Hfind Hsatisfied
             Hagrees Hvalue).
Qed.

Lemma add_equals_holds (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  import_holds import_var ->
  state_valid s ->
  envs_invariant s ->
  envs_kinded s ->
  state_holds s ->
  exists_in_target s var = true ->
  vector_holds s var kind simples ->
  vector_in_envs s simples ->
  vector_known s var simples ->
  (knows (Join_env.source_env s) var -> joined_everywhere s var) ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  later s s' ∧ state_holds s'.
Proof.
  intros Hfresh Himport Hvalid Hinvariant Hkinded Hholds Hvar Hvector Henvs
    Hknown Hlevels Hadd.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [canonical | ] eqn:Hcanonical; [ | discriminate ].
  pose proof (canonical_simples_cases s simples canonical Hcanonical)
    as Hcases.
  destruct canonical as [canonical | imported | canonicals].
  - destruct Hcases as [Hstands Hsource].
    apply (add_equation_holds s s' var (alias_type_of kind canonical) Hvalid
             Hholds); [ | exact Hadd ].
    apply (alias_equation_holds s var kind simples canonical Hvalid
             Hinvariant Hvar); [ | exact Hvector | exact Hstands ].
    intros x Hx. exact (source_exists _ _ (Hsource x Hx)).
  - destruct (import_var s imported) as [s1 | ] eqn:Hvar1; [ | discriminate ].
    destruct (Himport _ _ _ Hvalid Hinvariant Hkinded Hholds Hvar1)
      as [Hlater1 [Hexists1 Hholds1]].
    assert (Hequation :
              equation_holds s1 var (alias_type_of kind (Var imported))).
    { apply (alias_equation_holds s1 var kind simples (Var imported)
               (proj1 (proj2 Hlater1))
               (later_envs_invariant _ _ Hlater1 Hinvariant)
               (later_exists _ _ _ Hlater1 Hvar)).
      - intros x [<- | []]. exact Hexists1.
      - exact (later_vector_holds _ _ _ _ _ Hlater1 Hvar Hvector).
      - exact (later_stands_in_for _ _ _ _ Hlater1 Hcases). }
    destruct (add_equation_holds s1 s' var _ (proj1 (proj2 Hlater1)) Hholds1
                Hequation Hadd) as [Hlater' Hholds'].
    split; [ exact (later_trans _ _ _ Hlater1 Hlater') | exact Hholds' ].
  - subst canonicals. injection Hadd as <-.
    exact (add_alias_holds p s kind var simples Hfresh Hvalid Hholds Hvar
             Hvector Henvs Hknown Hlevels).
Qed.

(* The bindings of a map from indices, in order, are its bindings. *)
Lemma in_insert {A : Type} (b b0 : Index.t * A) (sorted : list (Index.t * A)) :
  In b (Index.insert b0 sorted) <-> b = b0 ∨ In b sorted.
Proof.
  induction sorted as [ | b' rest IH]; cbn [Index.insert].
  - cbn [In]. split; intros [H | []]; left; symmetry; exact H.
  - destruct (Nat.leb (fst b0) (fst b')).
    + cbn [In]. split; (intros [H | H]; [ left; symmetry; exact H | ]);
        right; exact H.
    + cbn [In]. rewrite IH. split.
      * intros [H | [H | H]]; [ right; left; exact H | left; exact H | ].
        right. right. exact H.
      * intros [H | [H | H]]; [ right; left; exact H | left; exact H | ].
        right. right. exact H.
Qed.

Lemma in_bindings {A : Type} (m : Keyed.finite_map Index.t A)
    (b : Index.t * A) :
  In b (Index.bindings m) <-> In b (Keyed.compact Index.equal m).
Proof.
  unfold Index.bindings.
  induction (Keyed.compact Index.equal m) as [ | b' l IH];
    cbn [List.fold_right].
  - reflexivity.
  - rewrite in_insert, IH. cbn [In].
    split; (intros [H | H]; [ left; symmetry; exact H | right; exact H ]).
Qed.

Lemma bindings_lookup {A : Type} (m : Keyed.finite_map Index.t A)
    (index : Index.t) (a : A) :
  In (index, a) (Index.bindings m)
  <-> Keyed.lookup Index.equal index m = Some a.
Proof.
  rewrite in_bindings. apply (keyed_in_compact Index.equal index_equal_decides).
Qed.

Lemma choose_lookup {A : Type} (m : Keyed.finite_map Index.t A)
    (index : Index.t) (a : A) :
  Index.choose m = Some (index, a) ->
  Keyed.lookup Index.equal index m = Some a.
Proof.
  unfold Index.choose. destruct (Index.bindings m) as [ | b l] eqn:Hbindings;
    [ discriminate | ].
  intros Hchoose. injection Hchoose as ->. apply bindings_lookup.
  rewrite Hbindings. left. reflexivity.
Qed.

Lemma concrete_holds (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  parameters_type_eqb_sound p ->
  import_holds import_var ->
  state_valid s ->
  envs_invariant s ->
  envs_kinded s ->
  state_holds s ->
  types_hold s var types ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  later s s' ∧ state_holds s'.
Proof.
  intros Heqb Himport Hvalid Hinvariant Hkinded Hholds Htypes Hconcrete.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index0 any_ty] | ] eqn:Hchoose;
    [ | discriminate ].
  destruct (forallb _ _) eqn:Hall.
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    assert (Hequation : equation_holds s var any_ty).
    { destruct Htypes as [Hvar [Hfound Htrue]].
      pose proof (choose_lookup _ _ _ Hchoose) as Hany.
      split; [ exact Hvar | ]. split.
      - intros y Hy. destruct (Hfound _ _ Hany) as [use0 [Huse0 [Hscope0 _]]].
        right. exists index0, use0.
        split; [ exact Huse0 | exact (Hscope0 y Hy) ].
      - intros index use statics valuation v Huse Hsatisfied _ Hv.
        destruct (Htrue _ _ _ _ _ Huse Hsatisfied Hv) as [t [Ht Hvt]].
        destruct (Hfound _ _ Ht) as [use' [Huse' [Hscope _]]].
        rewrite Huse in Huse'. injection Huse' as <-.
        pose proof (target_valuation_type s index use statics valuation v t
                      Hvalid Huse Hscope Hvt) as Htarget.
        rewrite forallb_forall in Hall.
        pose proof (Hall (index, t)
                      (proj2 (keyed_in_compact Index.equal index_equal_decides
                                index t types) Ht)) as Htest.
        cbn [fst snd] in Htest. apply orb_true_iff in Htest.
        destruct Htest as [Hsame | Hequal].
        + apply index_equal_decides in Hsame. subst index0.
          rewrite Hany in Ht. injection Ht as <-. exact Htarget.
        + exact (Heqb any_ty t statics _ v Hequal Htarget). }
    destruct (add_equation_holds s s1 var any_ty Hvalid Hholds Hequation Hadd)
      as [Hlater1 Hholds1].
    destruct (import_vars_holds import_var (free_variables any_ty) Himport s1
                s' (proj1 (proj2 Hlater1))
                (later_envs_invariant _ _ Hlater1 Hinvariant)
                (later_envs_kinded _ _ Hlater1 Hkinded) Hholds1 Hconcrete)
      as [Hlater' Hholds'].
    split; [ exact (later_trans _ _ _ Hlater1 Hlater') | exact Hholds' ].
  - injection Hconcrete as <-.
    exact (add_equations_holds s var types Hvalid Hholds Htypes).
Qed.

(* ---------------------------------------------------------------------- *)
(* Bringing a variable in                                                  *)
(* ---------------------------------------------------------------------- *)

Lemma invariant_types_closed (tenv : Typing_env.t) :
  typing_env_invariant tenv -> types_closed tenv.
Proof.
  intros [Hwf _]. apply typing_env_well_formed_iff in Hwf.
  unfold typing_env_conditions in Hwf. tauto.
Qed.

(* A variable that the source does not know and a joined environment
   does: where it has a value in the valuation of the target, the input
   knows it, and that is the value that it has there. *)
Lemma unmade_value (s : Join_env.t) (index : Index.t) (use : Typing_env.t)
    (valuation : env) (x : variable) (v : value) :
  state_valid s ->
  Join_env.joined_env s index = Some use ->
  ~ knows (Join_env.source_env s) x ->
  (∃ (index0 : Index.t) (use0 : Typing_env.t),
     Join_env.joined_env s index0 = Some use0 ∧ knows use0 x) ->
  lookup x (target_valuation s index use valuation) = Some v ->
  knows use x ∧ lookup x valuation = Some v.
Proof.
  intros Hvalid Huse Hunknown [index0 [use0 [Huse0 Hknown0]]] Hv.
  rewrite lookup_target_valuation in Hv.
  rewrite (known_no_definition s index0 index use0 valuation x Hvalid Huse0)
    in Hv.
  2:{ unfold known_to. unfold knows in Hknown0. rewrite Hknown0. reflexivity. }
  destruct (known_to s use x) eqn:Hknown; [ | discriminate ].
  split; [ | exact Hv ].
  unfold known_to in Hknown. apply orb_true_iff in Hknown.
  destruct Hknown as [Hknown | Hknown]; [ exact Hknown | ].
  exfalso. exact (Hunknown Hknown).
Qed.

Lemma types_of_imported_var_lookup
    (envs :
       list (Index.t * (Typing_env.t * Keyed.finite_map name flambda_type)))
    (var : variable) (kind : flambda_kind) :
  ∀ types : Keyed.finite_map Index.t flambda_type,
    Joined_envs.types_of_imported_var envs var kind = Some types ->
    NoDup (map fst envs) ->
    ∀ (index : Index.t) (env : Typing_env.t)
      (equations : Keyed.finite_map name flambda_type),
      (index, (env, equations)) ∈ envs ->
      knows env var ->
      ∃ ty : flambda_type,
        Typing_env.find env (Name_var var) (Some kind) = Some ty
        ∧ Keyed.lookup Index.equal index types = Some ty.
Proof.
  induction envs as [ | [index0 [env0 equations0]] envs IH];
    intros types H Hdistinct index env equations Hin Hknown; [ destruct Hin | ].
  cbn [Joined_envs.types_of_imported_var] in H.
  destruct (Joined_envs.types_of_imported_var envs var kind)
    as [types0 | ]; [ | discriminate ].
  cbn [map fst] in Hdistinct.
  inversion Hdistinct as [ | ? ? Hfresh Hdistinct0]; subst.
  assert (Hother :
            (index, (env, equations)) ∈ envs ->
            Index.equal index index0 = false).
  { intros Hin'. destruct (Index.equal index index0) eqn:Heq; [ | reflexivity ].
    exfalso. apply index_equal_decides in Heq. subst index0. apply Hfresh.
    apply in_map_iff. exists (index, (env, equations)).
    split; [ reflexivity | exact Hin' ]. }
  destruct (Typing_env.mem None env0 (Name_var var)) eqn:Hmem.
  - destruct (Typing_env.find env0 (Name_var var) (Some kind)) as [ty0 | ]
      eqn:Hfind; [ | discriminate ].
    injection H as <-. destruct Hin as [Heq | Hin].
    + injection Heq as <- <- <-. exists ty0. split; [ exact Hfind | ].
      rewrite keyed_lookup_cons.
      rewrite (proj2 (index_equal_decides index0 index0) eq_refl).
      reflexivity.
    + destruct (IH _ eq_refl Hdistinct0 _ _ _ Hin Hknown) as [ty [Hty Hlookup]].
      exists ty. split; [ exact Hty | ].
      rewrite keyed_lookup_cons, (Hother Hin). exact Hlookup.
  - injection H as <-. destruct Hin as [Heq | Hin].
    + injection Heq as <- <- <-. unfold knows in Hknown. congruence.
    + exact (IH _ eq_refl Hdistinct0 _ _ _ Hin Hknown).
Qed.

Lemma canonicals_of_alias_types_lookup (envs : Joined_envs.t) :
  ∀ (types : list (Index.t * flambda_type))
    (canonicals : Simples_in_joined_envs.t),
    Joined_envs.canonicals_of_alias_types envs types = Some (Some canonicals)
    ->
    ∀ (index : Index.t) (ty : flambda_type),
      Keyed.lookup Index.equal index types = Some ty ->
      ∃ (env : Typing_env.t) (alias : simple),
        Joined_envs.get_nth_joined_env envs index = Some env
        ∧ get_alias_opt ty = Some alias
        ∧ Simples_in_joined_envs.find index canonicals
          = Some (Typing_env.get_canonical_simple_ignoring_name_mode env alias).
Proof.
  induction types as [ | [index0 ty0] types IH];
    intros canonicals H index ty Hlookup; [ discriminate | ].
  cbn [Joined_envs.canonicals_of_alias_types] in H.
  destruct (Joined_envs.get_nth_joined_env envs index0) as [env0 | ] eqn:Henv0;
    [ | discriminate ].
  destruct (Joined_envs.canonicals_of_alias_types envs types)
    as [[canonicals0 | ] | ]; cbn in H; try discriminate.
  2:{ destruct (get_alias_opt ty0); discriminate. }
  destruct (get_alias_opt ty0) as [alias0 | ] eqn:Halias0; [ | discriminate ].
  injection H as <-.
  rewrite keyed_lookup_cons in Hlookup.
  unfold Simples_in_joined_envs.find. rewrite keyed_lookup_cons.
  destruct (Index.equal index index0) eqn:Heq.
  - injection Hlookup as <-. apply index_equal_decides in Heq. subst index0.
    exists env0, alias0. repeat split; [ exact Henv0 | exact Halias0 ].
  - exact (IH _ eq_refl _ _ Hlookup).
Qed.

Lemma get_alias_free_variables (ty : flambda_type) (alias : simple)
    (x : variable) :
  get_alias_opt ty = Some alias ->
  x ∈ free_variables_of_simple alias ->
  x ∈ free_variables ty.
Proof.
  destruct ty as [d | d | k d | d | d]; destruct d as [ | [head | s] | ];
    cbn; intros H; try discriminate; injection H as <-; intros Hx; exact Hx.
Qed.

(* The vector of the canonical elements of types that are aliases
   stands for a variable whose types they are. *)
Lemma canonicals_vector_holds (s : Join_env.t) (x : variable)
    (kind : flambda_kind) (types : Keyed.finite_map Index.t flambda_type)
    (canonicals : Simples_in_joined_envs.t) :
  types_hold s x types ->
  (∀ (index : Index.t) (t : flambda_type),
     Keyed.lookup Index.equal index types = Some t ->
     FlambdaTypeGrammar.kind t = kind) ->
  Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
    (Join_env.joined_envs s) types
  = Some (Some canonicals) ->
  vector_holds s x kind canonicals.
Proof.
  intros [_ [Hfound Htrue]] Hkinds Hcanonicals index use statics valuation v
    Huse Hsatisfied Hv.
  destruct (Htrue _ _ _ _ _ Huse Hsatisfied Hv) as [t [Ht Hvt]].
  split.
  { rewrite <- (Hkinds _ _ Ht). exact (type_satisfied_kind _ _ _ _ Hvt). }
  unfold Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
    in Hcanonicals.
  pose proof Ht as Ht'.
  rewrite <- (keyed_lookup_compact Index.t Index.equal index_equal_decides)
    in Ht'.
  destruct (canonicals_of_alias_types_lookup _ _ _ Hcanonicals _ _ Ht')
    as [env [alias [Henv [Halias Hfind]]]].
  change (Join_env.joined_env s index = Some env) in Henv.
  rewrite Huse in Henv. injection Henv as <-.
  exists alias. split; [ | split; [ exact Hfind | ] ].
  - destruct (Hfound _ _ Ht) as [use' [Huse' [Hscope _]]].
    rewrite Huse in Huse'. injection Huse' as <-.
    intros y Hy. exact (Hscope y (get_alias_free_variables _ _ _ Halias Hy)).
  - exact (proj1 (proj1 (get_alias_satisfied statics valuation v t alias
                           Halias) Hvt)).
Qed.

(* The vector of the canonical elements of types that are aliases is
   over the joined environments, if the aliases are in scope. *)
Lemma canonicals_in_envs (s : Join_env.t)
    (types : Keyed.finite_map Index.t flambda_type)
    (canonicals : Simples_in_joined_envs.t) :
  envs_invariant s ->
  (∀ (index : Index.t) (t : flambda_type),
     Keyed.lookup Index.equal index types = Some t ->
     ∃ use : Typing_env.t,
       Join_env.joined_env s index = Some use
       ∧ ∀ alias : simple,
           get_alias_opt t = Some alias -> simple_in_scope use alias) ->
  Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
    (Join_env.joined_envs s) types
  = Some (Some canonicals) ->
  vector_in_envs s canonicals.
Proof.
  intros Hinvariant Htypes Hcanonicals index simple Hfind.
  apply (keyed_lookup_in Index.equal index_equal_decides) in Hfind.
  destruct (canonicals_of_alias_types_spec _ _ _ Hcanonicals _ _ Hfind)
    as [ty [env [alias [Hin [Henv [Halias ->]]]]]].
  apply (keyed_in_compact Index.equal index_equal_decides) in Hin.
  destruct (Htypes _ _ Hin) as [use [Huse Hscope]].
  pose proof (eq_trans (eq_sym Henv) Huse) as Heq. injection Heq as ->.
  exists use. split; [ exact Huse | ].
  exact (canonical_in_scope use alias (Hinvariant _ _ Huse)
           (Hscope alias Halias)).
Qed.

Lemma types_hold_aliases (s : Join_env.t) (x : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  types_hold s x types ->
  ∀ (index : Index.t) (t : flambda_type),
    Keyed.lookup Index.equal index types = Some t ->
    ∃ use : Typing_env.t,
      Join_env.joined_env s index = Some use
      ∧ ∀ alias : simple,
          get_alias_opt t = Some alias -> simple_in_scope use alias.
Proof.
  intros [_ [Hfound _]] index t Ht.
  destruct (Hfound _ _ Ht) as [use [Huse [Hscope _]]].
  exists use. split; [ exact Huse | ].
  intros alias Halias y Hy.
  exact (Hscope y (get_alias_free_variables _ _ _ Halias Hy)).
Qed.

Lemma canonicals_of_map_lookup (envs : Joined_envs.t)
    (f : flambda_type -> flambda_type)
    (types : Keyed.finite_map Index.t flambda_type)
    (canonicals : Simples_in_joined_envs.t) (index : Index.t)
    (t : flambda_type) :
  Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn envs
    (Keyed.map_values f types)
  = Some (Some canonicals) ->
  Keyed.lookup Index.equal index types = Some t ->
  ∃ (env : Typing_env.t) (alias : simple),
    Joined_envs.get_nth_joined_env envs index = Some env
    ∧ get_alias_opt (f t) = Some alias
    ∧ Simples_in_joined_envs.find index canonicals
      = Some (Typing_env.get_canonical_simple_ignoring_name_mode env alias).
Proof.
  unfold Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn.
  intros Hcanonicals Ht.
  apply (canonicals_of_alias_types_lookup _ _ _ Hcanonicals).
  rewrite (keyed_lookup_compact Index.t Index.equal index_equal_decides),
    (keyed_lookup_map_values Index.t Index.equal), Ht.
  reflexivity.
Qed.

Lemma import_body_holds (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  import_keeps import_var ->
  import_holds import_var ->
  import_holds (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Hfresh Heqb Hkeeps Himport s s' x Hvalid Hinvariant Hkinded Hholds
    Hbody.
  destruct (import_body_keeps p import_var Hfresh Hkeeps s s' x Hvalid Hbody)
    as [_ [_ Hexists]].
  assert (Hgoal : later s s' ∧ state_holds s').
  2:{ destruct Hgoal as [Hlater' Hholds'].
      split; [ exact Hlater' | split; [ exact Hexists | exact Hholds' ] ]. }
  clear Hexists.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x)
    eqn:Hsource.
  { injection Hbody as <-.
    split; [ exact (later_refl s Hvalid) | exact Hholds ]. }
  destruct (among x (Join_env.imported_vars s)) eqn:Hamong.
  { injection Hbody as <-.
    split; [ exact (later_refl s Hvalid) | exact Hholds ]. }
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ] eqn:Hkind; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  assert (Hunknown : ~ knows (Join_env.source_env s) x).
  { unfold knows. unfold Source_env.exists_in_source_env in Hsource.
    rewrite Hsource. discriminate. }
  pose proof (kind_of_imported_var_known _ _ _ Hkind) as Hsomewhere.
  destruct (with_import_keeps s x kind Hvalid Hunknown Hsomewhere)
    as [Hvalid1 [Hgrows1 Hexists1]].
  assert (Hlater1 : later s (with_import s x kind)).
  { split; [ exact Hvalid | ]. split; [ exact Hvalid1 | ].
    split; [ exact Hgrows1 | ].
    unfold with_import, Join_env.source_env,
      Bindings_in_target_env.import_from_all_envs.
    cbn [Join_env.bindings].
    destruct (is_bound x
                (Bindings_in_target_env.imported_variables
                   (Join_env.bindings s)));
      reflexivity. }
  pose proof (later_holds s (with_import s x kind) Hlater1 eq_refl eq_refl
                eq_refl eq_refl eq_refl Hholds) as Hholds1.
  pose proof (later_envs_invariant _ _ Hlater1 Hinvariant) as Hinvariant1.
  pose proof (later_envs_kinded _ _ Hlater1 Hkinded) as Hkinded1.
  assert (Hunknown1 : ~ knows (Join_env.source_env (with_import s x kind)) x).
  { rewrite (proj2 (proj2 (proj2 Hlater1))). exact Hunknown. }
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ] eqn:Htypes;
    [ | discriminate ].
  destruct (types_of_imported_var_spec _ _ _ _ Htypes) as [_ Hfound].
  assert (Hx_types : types_hold (with_import s x kind) x types).
  { split; [ exact Hexists1 | ]. split.
    - intros index t Ht.
      apply (keyed_lookup_in Index.equal index_equal_decides) in Ht.
      destruct (Hfound _ _ Ht) as [env [equations [Hin Hfind]]].
      pose proof (in_compact_joined_env s index env equations Hin) as Huse.
      exists env. split; [ exact Huse | ]. split.
      + intros y Hy.
        exact (find_closed env (Name_var x) (Some kind) t y
                 (invariant_types_closed env (Hinvariant _ _ Huse)) Hfind Hy).
      + exact (find_kinded_in env _ _ t (Hkinded _ _ Huse) Hfind).
    - intros index use statics valuation v Huse Hsatisfied Hv.
      destruct (unmade_value (with_import s x kind) index use valuation x v
                  Hvalid1 Huse Hunknown1 Hsomewhere Hv) as [Hknown Hvalue].
      destruct (joined_env_in_compact s index use Huse) as [equations Hin].
      destruct (types_of_imported_var_lookup _ _ _ _ Htypes
                  (proj1 (keyed_distinct_NoDup Index.equal index_equal_decides
                            _)
                     (keyed_compact_distinct Index.t Index.equal
                        index_equal_decides _ _))
                  index use equations Hin Hknown) as [ty [Hfind Hlookup]].
      exists ty. split; [ exact Hlookup | ].
      exact (find_satisfied_strictly statics valuation use (Name_var x)
               (Some kind) ty v Hsatisfied Hfind Hvalue). }
  destruct (existsb _ _).
  - destruct
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         _ _) as [[canonical_simples | ] | ] eqn:Hcanonicals;
      try discriminate.
    assert (Hvector : vector_holds (with_import s x kind) x kind
                        canonical_simples).
    { intros index use statics valuation v Huse Hsatisfied Hv.
      destruct Hx_types as [_ [Hfound' Htrue]].
      destruct (Htrue _ _ _ _ _ Huse Hsatisfied Hv) as [t [Ht Hvt]].
      destruct (unmade_value (with_import s x kind) index use valuation x v
                  Hvalid1 Huse Hunknown1 Hsomewhere Hv) as [Hknown Hvalue].
      assert (Hkind_t : FlambdaTypeGrammar.kind t = kind).
      { pose proof Ht as Hin.
        apply (keyed_lookup_in Index.equal index_equal_decides) in Hin.
        destruct (Hfound _ _ Hin) as [env [equations [_ Hfind]]].
        exact (find_kind _ _ _ _ Hfind). }
      split; [ rewrite <- Hkind_t; exact (type_satisfied_kind _ _ _ _ Hvt) | ].
      destruct (canonicals_of_map_lookup _ _ _ _ _ _ Hcanonicals Ht)
        as [env [alias [Henv [Halias Hfind]]]].
      pose proof (eq_trans (eq_sym Henv) Huse) as Heq. injection Heq as ->.
      cbv beta in Halias. exists alias.
      destruct (get_alias_opt t) as [a | ] eqn:Ha; cbv beta iota in Halias.
      - rewrite Ha in Halias. injection Halias as <-.
        split; [ | split; [ exact Hfind | ] ].
        + destruct (Hfound' _ _ Ht) as [use' [Huse' [Hscope _]]].
          pose proof (eq_trans (eq_sym Huse') Huse) as Heq.
          injection Heq as ->.
          intros y Hy.
          exact (Hscope y (get_alias_free_variables _ _ _ Ha Hy)).
        + exact (proj1 (proj1 (get_alias_satisfied statics valuation v t a Ha)
                          Hvt)).
      - rewrite get_alias_alias_type_of in Halias. injection Halias as <-.
        split; [ | split; [ exact Hfind | exact Hvalue ] ].
        intros y [<- | []]. exact Hknown. }
    assert (Henvs : vector_in_envs (with_import s x kind) canonical_simples).
    { refine (canonicals_in_envs (with_import s x kind) _ canonical_simples
                Hinvariant1 _ Hcanonicals).
      intros index t' Ht'.
      rewrite (keyed_lookup_map_values Index.t Index.equal) in Ht'.
      destruct (Keyed.lookup Index.equal index types) as [t0 | ] eqn:Ht0;
        [ | discriminate ].
      injection Ht' as <-.
      destruct Hx_types as [_ [Hfound' _]].
      destruct (Hfound' _ _ Ht0) as [use [Huse [Hscope _]]].
      exists use. split; [ exact Huse | ].
      intros alias Halias. cbv beta in Halias.
      destruct (get_alias_opt t0) as [a | ] eqn:Ha; cbv beta iota in Halias.
      - rewrite Ha in Halias. injection Halias as <-.
        intros y Hy. exact (Hscope y (get_alias_free_variables _ _ _ Ha Hy)).
      - rewrite get_alias_alias_type_of in Halias. injection Halias as <-.
        apply (keyed_lookup_in Index.equal index_equal_decides) in Ht0.
        destruct (Hfound _ _ Ht0) as [env [equations [Hin Hfind]]].
        pose proof (in_compact_joined_env s index env equations Hin) as Huse0.
        pose proof (eq_trans (eq_sym Huse0) Huse) as Heq. injection Heq as ->.
        intros y [<- | []].
        exact (simple_of_kind_var_known use x kind (ex_intro _ t0 Hfind)). }
    assert (Hknown_x : vector_known (with_import s x kind) x
                         canonical_simples).
    { intros index c Hc.
      apply (keyed_lookup_in Index.equal index_equal_decides) in Hc.
      destruct (canonicals_of_alias_types_spec _ _ _ Hcanonicals _ _ Hc)
        as [ty' [env [alias [Hin' [Hnth _]]]]].
      exists env. split; [ exact Hnth | ].
      apply (keyed_in_compact Index.equal index_equal_decides) in Hin'.
      rewrite (keyed_lookup_map_values Index.t Index.equal) in Hin'.
      destruct (Keyed.lookup Index.equal index types) as [t0 | ] eqn:Ht0;
        [ | discriminate ].
      apply (keyed_lookup_in Index.equal index_equal_decides) in Ht0.
      destruct (Hfound _ _ Ht0) as [env0 [equations0 [Hin0 Hfind0]]].
      pose proof (in_compact_joined_env s index env0 equations0 Hin0)
        as Huse0.
      change (Join_env.joined_env s index = Some env) in Hnth.
      rewrite Hnth in Huse0. injection Huse0 as <-.
      exact (simple_of_kind_var_known env x kind (ex_intro _ t0 Hfind0)). }
    destruct (add_equals_holds p import_var (with_import s x kind) s' kind x
                canonical_simples Hfresh Himport Hvalid1 Hinvariant1 Hkinded1
                Hholds1 Hexists1 Hvector Henvs Hknown_x
                (fun Hk => False_ind _ (Hunknown1 Hk)) Hbody)
      as [Hlater' Hholds'].
    split; [ exact (later_trans _ _ _ Hlater1 Hlater') | exact Hholds' ].
  - destruct (concrete_holds p import_var (with_import s x kind) s' x types
                Heqb Himport Hvalid1 Hinvariant1 Hkinded1 Hholds1 Hx_types
                Hbody) as [Hlater' Hholds'].
    split; [ exact (later_trans _ _ _ Hlater1 Hlater') | exact Hholds' ].
Qed.

Lemma import_var_holds (p : Join_env.parameters) (fuel : nat) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  import_holds (Join_env.import_var_transitive p fuel).
Proof.
  intros Hfresh Heqb. induction fuel as [ | fuel IH].
  - intros s s' x _ _ _ _ H. discriminate.
  - exact (import_body_holds p _ Hfresh Heqb (import_var_keeps p fuel Hfresh)
             IH).
Qed.

(* ---------------------------------------------------------------------- *)
(* The three functions that the join of types calls                        *)
(* ---------------------------------------------------------------------- *)

Lemma import_type_holds (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (ty : flambda_type) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  state_valid s ->
  envs_invariant s ->
  envs_kinded s ->
  state_holds s ->
  Join_env.import_type p fuel s ty = Some s' ->
  later s s' ∧ state_holds s'.
Proof.
  intros Hfresh Heqb Hvalid Hinvariant Hkinded Hholds Himport.
  exact (import_vars_holds _ (free_variables ty)
           (import_var_holds p fuel Hfresh Heqb) s s' Hvalid Hinvariant
           Hkinded Hholds Himport).
Qed.

Lemma n_way_join_simples_holds (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (kind : flambda_kind)
    (simples : list (Index.t * simple)) (result : Or_bottom.t simple) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  state_valid s ->
  envs_invariant s ->
  envs_kinded s ->
  state_holds s ->
  join_args_distinct simples ->
  join_args_in_scope Join_env.joined_env simple_in_scope s simples ->
  Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
  later s s' ∧ state_holds s'.
Proof.
  intros Hfresh Heqb Hvalid Hinvariant Hkinded Hholds Hdistinct Hscope Hjoin.
  unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as _ <-.
    split; [ exact (later_refl s Hvalid) | exact Hholds ]. }
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode _ _)
    as [canonicals | ] eqn:Hcanonicals; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [[simple0 | var | canonicals'] | ] eqn:Hcanonical;
    [ | | | discriminate ].
  - injection Hjoin as _ <-.
    split; [ exact (later_refl s Hvalid) | exact Hholds ].
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as _ <-.
    destruct (import_var_holds p fuel Hfresh Heqb _ _ _ Hvalid Hinvariant
                Hkinded Hholds Himport) as [Hlater' [_ Hholds']].
    split; [ exact Hlater' | exact Hholds' ].
  - destruct (Join_env.existential_for_these_canonicals p s canonicals' kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as _ <-.
    refine (existential_holds _ _ _ _ _ _ Hfresh Hvalid Hholds _ Hmake).
    pose proof (canonical_simples_cases s canonicals _ Hcanonical) as Hsame.
    cbn in Hsame. subst canonicals'.
    intros index canonical Hfind.
    destruct (canonical_simples_binding _ _ _ _ _ Hcanonicals Hfind)
      as [simple Hin].
    destruct (canonical_simples_find _ _ _ _ _ Hcanonicals Hdistinct Hin)
      as [env [Henv Hfind']].
    destruct (Hscope _ _ Hin) as [use [Huse Hsimple]].
    pose proof (eq_trans (eq_sym Henv) Huse) as Heq. injection Heq as ->.
    pose proof (eq_trans (eq_sym Hfind) Hfind') as Heq. injection Heq as ->.
    exists use. split; [ exact Huse | ].
    exact (canonical_in_scope use simple (Hinvariant _ _ Huse) Hsimple).
Qed.

Lemma import_equations_holds (p : Join_env.parameters) (fuel : nat) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  ∀ (equations : list (name * flambda_type)) (s s' : Join_env.t),
    state_valid s ->
    envs_invariant s ->
    envs_kinded s ->
    state_holds s ->
    List.fold_left
      (fun env equation =>
         let* env := env in
         let* env :=
           Join_env.import_name_transitive p fuel env (fst equation)
         in
         Join_env.import_type_transitive p fuel env (snd equation))
      equations (Some s)
    = Some s' ->
    later s s' ∧ state_holds s'.
Proof.
  intros Hfresh Heqb.
  induction equations as [ | [n ty] equations IH];
    intros s s' Hvalid Hinvariant Hkinded Hholds Hfold.
  - injection Hfold as <-.
    split; [ exact (later_refl s Hvalid) | exact Hholds ].
  - cbn [List.fold_left fst snd] in Hfold.
    destruct (Join_env.import_name_transitive p fuel s n) as [s1 | ]
      eqn:Hname.
    + assert (Hfirst : later s s1 ∧ state_holds s1).
      { destruct n as [x | sym].
        - destruct (import_var_holds p fuel Hfresh Heqb _ _ _ Hvalid
                      Hinvariant Hkinded Hholds Hname) as [Hl [_ Hh]].
          split; [ exact Hl | exact Hh ].
        - injection Hname as <-.
          split; [ exact (later_refl s Hvalid) | exact Hholds ]. }
      destruct Hfirst as [Hlater1 Hholds1].
      destruct (Join_env.import_type_transitive p fuel s1 ty) as [s2 | ]
        eqn:Htype.
      * destruct (import_type_holds p fuel s1 s2 ty Hfresh Heqb
                    (proj1 (proj2 Hlater1))
                    (later_envs_invariant _ _ Hlater1 Hinvariant)
                    (later_envs_kinded _ _ Hlater1 Hkinded) Hholds1 Htype)
          as [Hlater2 Hholds2].
        pose proof (later_trans _ _ _ Hlater1 Hlater2) as Hlater12.
        destruct (IH _ _ (proj1 (proj2 Hlater12))
                    (later_envs_invariant _ _ Hlater12 Hinvariant)
                    (later_envs_kinded _ _ Hlater12 Hkinded) Hholds2 Hfold)
          as [Hlater' Hholds'].
        split; [ exact (later_trans _ _ _ Hlater12 Hlater') | exact Hholds' ].
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
    + exfalso. clear -Hfold.
      induction equations as [ | e equations IH];
        [ discriminate | exact (IH Hfold) ].
Qed.

Lemma n_way_join_env_extension_holds (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (extensions : list (Index.t * env_extension))
    (joined : env_extension) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  state_valid s ->
  envs_invariant s ->
  envs_kinded s ->
  state_holds s ->
  Join_env.n_way_join_env_extension p fuel s extensions
  = Some (Or_bottom.Ok (joined, s')) ->
  later s s' ∧ state_holds s'.
Proof.
  intros Hfresh Heqb Hvalid Hinvariant Hkinded Hholds Hjoin.
  unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as _ <-.
    exact (import_equations_holds p fuel Hfresh Heqb _ _ _ Hvalid Hinvariant
             Hkinded Hholds Hfold).
  - injection Hjoin as _ <-.
    split; [ exact (later_refl s Hvalid) | exact Hholds ].
Qed.

(* ---------------------------------------------------------------------- *)
(* The aliases of the variables of the source                              *)
(* ---------------------------------------------------------------------- *)

Lemma found_satisfied (statics : static_env) (valuation : env)
    (use : Typing_env.t) (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) (v : value) :
  typing_env_satisfied_strictly statics valuation use ->
  Keyed.lookup name_eqb n (Typing_env.names_to_types use)
  = Some (ty, binding_time_and_mode) ->
  value_of_name valuation n = Some v ->
  type_satisfied statics valuation v ty.
Proof.
  intros [_ [Hnames _]] Hlookup Hv.
  destruct (Hnames n ty binding_time_and_mode v Hlookup Hv) as [[[] _] | H].
  exact H.
Qed.

Lemma bound_knows (tenv : Typing_env.t) (x : variable) :
  Keyed.is_bound name_eqb (Name_var x) (Typing_env.names_to_types tenv) = true
  -> knows tenv x.
Proof.
  unfold knows, Typing_env.mem, Keyed.is_bound.
  destruct (Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv))
    as [[ty btm] | ]; [ reflexivity | discriminate ].
Qed.

Lemma types_of_name_lookup (envs : Joined_envs.t) (n : name)
    (index : Index.t) :
  Keyed.lookup Index.equal index (Join_env.types_of_name_in_levels envs n)
  = match Keyed.lookup Index.equal index envs with
    | Some (_, equations) => Keyed.lookup name_eqb n equations
    | None => None
    end.
Proof.
  unfold Join_env.types_of_name_in_levels.
  rewrite <- (keyed_lookup_compact Index.t Index.equal index_equal_decides _
                index envs).
  pose proof (proj1 (keyed_distinct_NoDup Index.equal index_equal_decides _)
                (keyed_compact_distinct Index.t Index.equal
                   index_equal_decides _ envs)) as Hdistinct.
  revert Hdistinct. generalize (Keyed.compact Index.equal envs) as l.
  induction l as [ | [index0 [use0 equations0]] l IH]; intros Hdistinct;
    [ reflexivity | ].
  cbn [flat_map fst snd]. cbn [map fst] in Hdistinct.
  inversion Hdistinct as [ | ? ? Hfresh Hdistinct0]; subst.
  rewrite (keyed_lookup_cons Index.equal index index0 (use0, equations0) l).
  destruct (Index.equal index index0) eqn:Heq.
  - apply index_equal_decides in Heq. subst index0.
    destruct (Keyed.lookup name_eqb n equations0) as [ty | ] eqn:Hn;
      cbn [app].
    + rewrite keyed_lookup_cons.
      rewrite (proj2 (index_equal_decides index index) eq_refl). reflexivity.
    + rewrite (IH Hdistinct0).
      destruct (Keyed.lookup Index.equal index l) as [[use eqs] | ] eqn:Hl;
        [ | reflexivity ].
      exfalso. apply Hfresh.
      apply (keyed_lookup_in Index.equal index_equal_decides) in Hl.
      apply in_map_iff. exists (index, (use, eqs)).
      split; [ reflexivity | exact Hl ].
  - destruct (Keyed.lookup name_eqb n equations0) as [ty | ]; cbn [app].
    + rewrite keyed_lookup_cons, Heq. exact (IH Hdistinct0).
    + exact (IH Hdistinct0).
Qed.

Lemma names_everywhere_bound (envs : Joined_envs.t) (n : name) :
  n ∈ Join_env.names_with_equations_everywhere envs ->
  ∀ (index : Index.t) (use : Typing_env.t)
    (equations : Keyed.finite_map name flambda_type),
    Keyed.lookup Index.equal index envs = Some (use, equations) ->
    Keyed.is_bound name_eqb n equations = true.
Proof.
  unfold Join_env.names_with_equations_everywhere.
  intros Hin index use equations Hlookup.
  apply (keyed_in_compact Index.equal index_equal_decides) in Hlookup.
  destruct (Keyed.compact Index.equal envs)
    as [ | [index0 [use0 equations0]] others]; [ destruct Hin | ].
  apply filter_In in Hin. destruct Hin as [Hdomain Hall].
  destruct Hlookup as [Heq | Hin'].
  - injection Heq as <- <- <-.
    rewrite <- (keyed_among_domain name name_eqb name_eqb_decides_equality).
    apply (keyed_among_in name name_eqb name_eqb_decides_equality).
    exact Hdomain.
  - rewrite forallb_forall in Hall. exact (Hall _ Hin').
Qed.

Lemma joined_env_lookup (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) :
  Join_env.joined_env s index = Some use
  <-> ∃ equations : Keyed.finite_map name flambda_type,
        Keyed.lookup Index.equal index (Join_env.joined_envs s)
        = Some (use, equations).
Proof.
  unfold Join_env.joined_env, Joined_envs.get_nth_joined_env.
  destruct (Keyed.lookup Index.equal index (Join_env.joined_envs s))
    as [[use' equations] | ]; cbn [option_map fst]; split.
  - intros H. injection H as ->. exists equations. reflexivity.
  - intros [equations' H]. injection H as -> _. reflexivity.
  - discriminate.
  - intros [equations' H]. discriminate.
Qed.

(* The types that the levels have for a variable of the source that
   has an equation in each are true of it. *)
Lemma level_types_hold (s : Join_env.t) (x : variable) :
  state_valid s ->
  envs_invariant s ->
  envs_kinded s ->
  levels_found (Join_env.joined_envs s) ->
  knows (Join_env.source_env s) x ->
  Name_var x ∈ Join_env.names_with_equations_everywhere (Join_env.joined_envs s)
  ->
  types_hold s x
    (Join_env.types_of_name_in_levels (Join_env.joined_envs s) (Name_var x)).
Proof.
  intros Hvalid Hinvariant Hkinded Hlevels Hknown Heverywhere.
  split; [ exact (source_exists s x Hknown) | ]. split.
  - intros index t Ht. rewrite types_of_name_lookup in Ht.
    destruct (Keyed.lookup Index.equal index (Join_env.joined_envs s))
      as [[use equations] | ] eqn:Hindex; [ | discriminate ].
    destruct (Hlevels _ _ _ _ _ Hindex Ht) as [btm Hfound].
    assert (Huse : Join_env.joined_env s index = Some use).
    { apply joined_env_lookup. exists equations. exact Hindex. }
    exists use. split; [ exact Huse | ]. split.
    + intros y Hy. apply bound_knows.
      exact (invariant_types_closed use (Hinvariant _ _ Huse) _ _ _ y Hfound
               Hy).
    + exact (proj1 (Hkinded _ _ Huse) _ _ _ Hfound).
  - intros index use statics valuation v Huse Hsatisfied Hv.
    rewrite (target_valuation_known s index use valuation x Hvalid Huse) in Hv.
    2:{ unfold known_to. unfold knows in Hknown. rewrite Hknown.
        apply orb_true_r. }
    apply joined_env_lookup in Huse. destruct Huse as [equations Hindex].
    pose proof (names_everywhere_bound _ _ Heverywhere _ _ _ Hindex) as Hbound.
    apply keyed_is_bound_lookup in Hbound. destruct Hbound as [t Ht].
    exists t. split; [ rewrite types_of_name_lookup, Hindex; exact Ht | ].
    destruct (Hlevels _ _ _ _ _ Hindex Ht) as [btm Hfound].
    exact (found_satisfied statics valuation use (Name_var x) t btm v
             Hsatisfied Hfound Hv).
Qed.

(* Those types are of one kind, if the state is kinded. *)
Lemma level_types_one_kind (s : Join_env.t) (x : variable) :
  state_kinded s ->
  levels_found (Join_env.joined_envs s) ->
  knows (Join_env.source_env s) x ->
  ∀ (index index' : Index.t) (t t' : flambda_type),
    Keyed.lookup Index.equal index
      (Join_env.types_of_name_in_levels (Join_env.joined_envs s) (Name_var x))
    = Some t ->
    Keyed.lookup Index.equal index'
      (Join_env.types_of_name_in_levels (Join_env.joined_envs s) (Name_var x))
    = Some t' ->
    FlambdaTypeGrammar.kind t = FlambdaTypeGrammar.kind t'.
Proof.
  intros [Hsource _] Hlevels Hknown.
  unfold knows, Typing_env.mem in Hknown.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types (Join_env.source_env s)))
    as [[ty btm] | ] eqn:Hx; [ | discriminate ].
  assert (Hone :
            ∀ (index : Index.t) (t : flambda_type),
              Keyed.lookup Index.equal index
                (Join_env.types_of_name_in_levels (Join_env.joined_envs s)
                   (Name_var x))
              = Some t ->
              FlambdaTypeGrammar.kind ty = FlambdaTypeGrammar.kind t).
  { intros index t Ht. rewrite types_of_name_lookup in Ht.
    destruct (Keyed.lookup Index.equal index (Join_env.joined_envs s))
      as [[use equations] | ] eqn:Hindex; [ | discriminate ].
    destruct (Hlevels _ _ _ _ _ Hindex Ht) as [btm' Hfound].
    apply (Hsource index use x (FlambdaTypeGrammar.kind t) ty btm);
      [ | | exact Hx ].
    - apply joined_env_lookup. exists equations. exact Hindex.
    - apply (simple_of_kind_name_iff use (Name_var x)). left.
      exists t, btm'. split; [ exact Hfound | reflexivity ]. }
  intros index index' t t' Ht Ht'.
  rewrite <- (Hone _ _ Ht). exact (Hone _ _ Ht').
Qed.

Lemma import_or_delay_holds (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  state_valid s ->
  envs_invariant s ->
  envs_kinded s ->
  state_holds s ->
  types_hold s var types ->
  (∀ (index index' : Index.t) (t t' : flambda_type),
     Keyed.lookup Index.equal index types = Some t ->
     Keyed.lookup Index.equal index' types = Some t' ->
     FlambdaTypeGrammar.kind t = FlambdaTypeGrammar.kind t') ->
  (∀ (index : Index.t) (t : flambda_type) (use : Typing_env.t),
     Keyed.lookup Index.equal index types = Some t ->
     Join_env.joined_env s index = Some use ->
     knows use var) ->
  (knows (Join_env.source_env s) var -> joined_everywhere s var) ->
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  later s s' ∧ state_holds s'.
Proof.
  intros Hfresh Heqb Hvalid Hinvariant Hkinded Hholds Htypes Hkinds Hknown
    Hlevels Himport.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in Himport.
  destruct (Index.choose types) as [[index0 any_ty] | ] eqn:Hchoose;
    [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
       (Join_env.joined_envs s) types)
    as [[canonicals | ] | ] eqn:Hcanonicals; [ | | discriminate ].
  - apply (add_equals_holds p _ s s' (FlambdaTypeGrammar.kind any_ty) var
             canonicals Hfresh (import_var_holds p fuel Hfresh Heqb) Hvalid
             Hinvariant Hkinded Hholds (proj1 Htypes));
      [ | | | exact Hlevels | exact Himport ].
    + apply (canonicals_vector_holds s var (FlambdaTypeGrammar.kind any_ty)
               types canonicals Htypes); [ | exact Hcanonicals ].
      intros index t Ht.
      exact (Hkinds index index0 t any_ty Ht (choose_lookup _ _ _ Hchoose)).
    + exact (canonicals_in_envs s types canonicals Hinvariant
               (types_hold_aliases s var types Htypes) Hcanonicals).
    + intros index c Hc.
      apply (keyed_lookup_in Index.equal index_equal_decides) in Hc.
      destruct (canonicals_of_alias_types_spec _ _ _ Hcanonicals _ _ Hc)
        as [ty [env [alias [Hin [Hnth _]]]]].
      exists env. split; [ exact Hnth | ].
      apply (keyed_in_compact Index.equal index_equal_decides) in Hin.
      exact (Hknown index ty env Hin Hnth).
  - exact (concrete_holds p _ s s' var types Heqb
             (import_var_holds p fuel Hfresh Heqb) Hvalid Hinvariant Hkinded
             Hholds Htypes Himport).
Qed.

Lemma join_aliases_fold (p : Join_env.parameters) (fuel : nat)
    (s0 : Join_env.t) (envs : Joined_envs.t) (exists0 : variable -> bool) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  envs_invariant s0 ->
  envs_kinded s0 ->
  ∀ (names : list name) (s s' : Join_env.t),
    (∀ x : variable,
       Name_var x ∈ names ->
       exists0 x = true ->
       types_hold s0 x (Join_env.types_of_name_in_levels envs (Name_var x))
       ∧ (∀ (index index' : Index.t) (t t' : flambda_type),
            Keyed.lookup Index.equal index
              (Join_env.types_of_name_in_levels envs (Name_var x))
            = Some t ->
            Keyed.lookup Index.equal index'
              (Join_env.types_of_name_in_levels envs (Name_var x))
            = Some t' ->
            FlambdaTypeGrammar.kind t = FlambdaTypeGrammar.kind t')
       ∧ (∀ (index : Index.t) (t : flambda_type) (use : Typing_env.t),
            Keyed.lookup Index.equal index
              (Join_env.types_of_name_in_levels envs (Name_var x))
            = Some t ->
            Join_env.joined_env s0 index = Some use ->
            knows use x)
       ∧ (knows (Join_env.source_env s0) x -> joined_everywhere s0 x)) ->
    later s0 s ->
    same_envs s0 s ->
    state_holds s ->
    List.fold_left
      (fun env name =>
         let* env := env in
         match name with
         | Name_symbol _ => Some env
         | Name_var var =>
           if exists0 var
           then
             Join_env.import_or_delay_n_way_join_type p fuel env var
               (Join_env.types_of_name_in_levels envs name)
           else Some env
         end)
      names (Some s)
    = Some s' ->
    later s s' ∧ state_holds s'.
Proof.
  intros Hfresh Heqb Hinvariant0 Hkinded0.
  induction names as [ | n names IH];
    intros s s' Hnames Hlater Hsame Hholds Hfold.
  - injection Hfold as <-.
    split; [ exact (later_refl _ (proj1 (proj2 Hlater))) | exact Hholds ].
  - cbn [List.fold_left] in Hfold.
    pose proof (fun x Hx => Hnames x (or_intror Hx)) as Hrest.
    destruct n as [x | sym];
      [ | exact (IH s s' Hrest Hlater Hsame Hholds Hfold) ].
    destruct (exists0 x) eqn:Hexists;
      [ | exact (IH s s' Hrest Hlater Hsame Hholds Hfold) ].
    destruct (Join_env.import_or_delay_n_way_join_type p fuel s x
                (Join_env.types_of_name_in_levels envs (Name_var x)))
      as [s1 | ] eqn:Hs1.
    + destruct (Hnames x (or_introl eq_refl) Hexists)
        as [Htypes [Hkinds [Hknown Hlevels]]].
      assert (Hknown_s :
                ∀ (index : Index.t) (t : flambda_type) (use : Typing_env.t),
                  Keyed.lookup Index.equal index
                    (Join_env.types_of_name_in_levels envs (Name_var x))
                  = Some t ->
                  Join_env.joined_env s index = Some use ->
                  knows use x).
      { intros index t use Ht Huse. rewrite (later_env _ _ _ Hlater) in Huse.
        exact (Hknown index t use Ht Huse). }
      assert (Hlevels_s :
                knows (Join_env.source_env s) x -> joined_everywhere s x).
      { intros Hk index use equations Hlookup.
        rewrite (proj2 (proj2 (proj2 Hlater))) in Hk.
        rewrite Hsame in Hlookup.
        exact (Hlevels Hk index use equations Hlookup). }
      destruct (import_or_delay_holds p fuel s s1 x _ Hfresh Heqb
                  (proj1 (proj2 Hlater))
                  (later_envs_invariant _ _ Hlater Hinvariant0)
                  (later_envs_kinded _ _ Hlater Hkinded0) Hholds
                  (later_types_hold _ _ _ _ Hlater Htypes) Hkinds Hknown_s
                  Hlevels_s Hs1)
        as [Hlater1 Hholds1].
      destruct (IH s1 s' Hrest (later_trans _ _ _ Hlater Hlater1)
                  (same_envs_trans _ _ _ Hsame
                     (import_or_delay_same_envs _ _ _ _ _ _ Hs1))
                  Hholds1 Hfold) as [Hlater' Hholds'].
      split; [ exact (later_trans _ _ _ Hlater1 Hlater') | exact Hholds' ].
    + exfalso. clear -Hfold.
      induction names as [ | n' names IH]; [ discriminate | exact (IH Hfold) ].
Qed.

(* [join_aliases_into_bindings], from a state whose variables are
   those of the source. *)
Lemma join_aliases_holds (p : Join_env.parameters) (fuel : nat)
    (s0 s' : Join_env.t) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  state_valid s0 ->
  state_kinded s0 ->
  envs_invariant s0 ->
  envs_kinded s0 ->
  state_holds s0 ->
  levels_found (Join_env.joined_envs s0) ->
  (∀ x : variable,
     exists_in_target s0 x = true -> knows (Join_env.source_env s0) x) ->
  Join_env.join_aliases_into_bindings p fuel s0 (Join_env.joined_envs s0)
  = Some s' ->
  later s0 s' ∧ state_holds s'.
Proof.
  intros Hfresh Heqb Hvalid0 Hstate0 Hinvariant0 Hkinded0 Hholds0 Hlevels
    Hsource Hjoin.
  apply (join_aliases_fold p fuel s0 (Join_env.joined_envs s0)
           (Bindings_in_target_env.exists_in_target_env (Join_env.bindings s0))
           Hfresh Heqb Hinvariant0 Hkinded0
           (Join_env.names_with_equations_everywhere (Join_env.joined_envs s0))
           s0 s');
    [ | exact (later_refl s0 Hvalid0) | reflexivity | exact Hholds0
      | exact Hjoin ].
  intros x Hx Hexists. rewrite <- kind_in_target_env_exists in Hexists.
  pose proof (Hsource x Hexists) as Hknown.
  split; [ | split; [ | split ] ].
  - exact (level_types_hold s0 x Hvalid0 Hinvariant0 Hkinded0 Hlevels Hknown
             Hx).
  - exact (level_types_one_kind s0 x Hstate0 Hlevels Hknown).
  - intros index t use Ht Huse. rewrite types_of_name_lookup in Ht.
    destruct (Keyed.lookup Index.equal index (Join_env.joined_envs s0))
      as [[use0 equations] | ] eqn:Hindex; [ | discriminate ].
    destruct (Hlevels _ _ _ _ _ Hindex Ht) as [btm Hfound].
    assert (Huse0 : Join_env.joined_env s0 index = Some use0).
    { apply joined_env_lookup. exists equations. exact Hindex. }
    rewrite Huse in Huse0. injection Huse0 as <-.
    apply bound_knows. unfold Keyed.is_bound. rewrite Hfound. reflexivity.
  - intros _ index use equations Hindex.
    exact (names_everywhere_bound _ _ Hx _ _ _ Hindex).
Qed.

(* ---------------------------------------------------------------------- *)
(* The invariant with what the state means, and the contracts with it      *)
(* ---------------------------------------------------------------------- *)

Lemma later_true (source : Typing_env.t) (s s' : Join_env.t) :
  later s s' ->
  state_holds s' ->
  Join_env.source_env s = source ->
  state_true source s'.
Proof.
  intros Hlater Hholds' Hsource. split; [ exact (proj1 (proj2 Hlater)) | ].
  split; [ exact Hholds' | ].
  rewrite (proj2 (proj2 (proj2 Hlater))). exact Hsource.
Qed.

Lemma state_true_witnesses (source : Typing_env.t) :
  valid_states_have_witnesses Join_env.joined_env
    Join_env.definitions_of_existentials (state_true source).
Proof.
  intros s index use statics valuation [Hvalid _] Huse Hsatisfied.
  exact (state_valid_has_witnesses s index use statics valuation Hvalid Huse
           Hsatisfied).
Qed.

Lemma n_way_join_simples_true_at (p : Join_env.parameters) (fuel : nat)
    (source : Typing_env.t) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  FlambdaJoinSpec.n_way_join_simples_sound typing_env_invariant
    Join_env.joined_env Join_env.definitions_of_existentials
    kind_in_target_env (state_true source) state_kinded
    (Join_env.n_way_join_simples p fuel).
Proof.
  intros Hfresh Heqb s s' kind simples result [Hvalid [Hholds Hsource]] Hwf
    Hkinded Hdistinct Hscope Hjoin.
  destruct (FlambdaJoinEnvProofs.n_way_join_simples_sound p fuel Hfresh s s'
              kind simples result Hvalid Hwf Hkinded Hdistinct Hscope Hjoin)
    as [Hvalid' Hrest].
  assert (Hscope' :
            join_args_in_scope Join_env.joined_env simple_in_scope s simples).
  { intros index simple Hin.
    destruct (Hscope index simple Hin) as [use [Huse Hkind]].
    exists use.
    split; [ exact Huse | ].
    exact (simple_of_kind_in_scope use simple kind Hkind). }
  destruct (n_way_join_simples_holds p fuel s s' kind simples result Hfresh
              Heqb Hvalid Hwf Hkinded Hholds Hdistinct Hscope' Hjoin)
    as [Hlater Hholds'].
  split; [ exact (later_true source s s' Hlater Hholds' Hsource) | ].
  exact Hrest.
Qed.

Lemma import_type_true_at (p : Join_env.parameters) (fuel : nat)
    (source : Typing_env.t) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  FlambdaJoinSpec.import_type_sound typing_env_invariant Join_env.joined_env
    Join_env.definitions_of_existentials kind_in_target_env
    (state_true source) state_kinded (Join_env.import_type p fuel).
Proof.
  intros Hfresh Heqb s s' ty [Hvalid [Hholds Hsource]] Hwf Hkinded Himport.
  destruct (FlambdaJoinEnvProofs.import_type_sound p fuel Hfresh s s' ty
              Hvalid Hwf Hkinded Himport) as [Hvalid' Hrest].
  destruct (import_type_holds p fuel s s' ty Hfresh Heqb Hvalid Hwf Hkinded
              Hholds Himport) as [Hlater Hholds'].
  split; [ exact (later_true source s s' Hlater Hholds' Hsource) | ].
  exact Hrest.
Qed.

Lemma n_way_join_env_extension_true_at (p : Join_env.parameters) (fuel : nat)
    (source : Typing_env.t) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  env_extension_eqb_sound p ->
  FlambdaJoinSpec.n_way_join_env_extension_sound typing_env_invariant
    Join_env.joined_env Join_env.definitions_of_existentials
    kind_in_target_env (state_true source) state_kinded
    (Join_env.n_way_join_env_extension p fuel).
Proof.
  intros Hfresh Heqb Hext s extensions result [Hvalid [Hholds Hsource]] Hwf
    Hkinded Hdistinct Hscope Hjoin.
  destruct (FlambdaJoinEnvProofs.n_way_join_env_extension_sound p fuel Hfresh
              Hext s extensions result Hvalid Hwf Hkinded Hdistinct Hscope
              Hjoin) as [Hbottom [Hok Hkinds]].
  split; [ exact Hbottom | ]. split; [ | exact Hkinds ].
  intros joined s' Hresult.
  destruct (Hok joined s' Hresult) as [Hvalid' Hrest].
  subst result.
  destruct (n_way_join_env_extension_holds p fuel s s' extensions joined
              Hfresh Heqb Hvalid Hwf Hkinded Hholds Hjoin)
    as [Hlater Hholds'].
  split; [ exact (later_true source s s' Hlater Hholds' Hsource) | ].
  exact Hrest.
Qed.


(* ---------------------------------------------------------------------- *)
(* The statements                                                          *)
(* ---------------------------------------------------------------------- *)

Theorem target_valuation_extends : target_valuation_extends_statement.
Proof. exact target_valuation_extends_at. Qed.

Theorem created_state_true : created_state_true_statement.
Proof.
  intros avoid source envs Hsource Huses.
  split; [ exact (created_state_valid avoid source envs Hsource Huses) | ].
  split; [ | reflexivity ].
  split; [ intros x ty Hx; cbn in Hx; discriminate | ].
  split; [ intros x types Hx; cbn in Hx; discriminate | ].
  split; [ intros e x Hx; cbn in Hx; destruct Hx | ].
  intros x simples kind Hx. cbn in Hx. discriminate.
Qed.

Theorem state_true_has_witnesses : state_true_has_witnesses_statement.
Proof. exact state_true_witnesses. Qed.

Theorem n_way_join_simples_true : n_way_join_simples_true_statement.
Proof. exact n_way_join_simples_true_at. Qed.

Theorem import_type_true : import_type_true_statement.
Proof. exact import_type_true_at. Qed.

Theorem n_way_join_env_extension_true
  : n_way_join_env_extension_true_statement.
Proof. exact n_way_join_env_extension_true_at. Qed.

Theorem join_aliases_true : join_aliases_true_statement.
Proof.
  intros p fuel source s0 s' Hfresh Heqb [Hvalid [Hholds Hsource]] Hstate
    Hinvariant Hkinded Hlevels Hknown Hjoin.
  assert (Hknown' :
            ∀ x : variable,
              exists_in_target s0 x = true ->
              knows (Join_env.source_env s0) x).
  { intros x Hx. rewrite Hsource. exact (Hknown x Hx). }
  destruct (join_aliases_holds p fuel s0 s' Hfresh Heqb Hvalid Hstate
              Hinvariant Hkinded Hholds Hlevels Hknown' Hjoin)
    as [Hlater Hholds'].
  split; [ exact (later_true source s0 s' Hlater Hholds' Hsource) | ].
  exact (proj1 (proj2 (proj2 Hlater))).
Qed.

(* ---------------------------------------------------------------------- *)
(* Recovering the inverse relations                                        *)
(* ---------------------------------------------------------------------- *)

(* The names that an extension makes aliases of one another, from a
   name. *)
Inductive alias_chain (extension : env_extension) : name -> name -> Prop :=
  | Chain_here (n : name) : alias_chain extension n n
  | Chain_step (n n' n'' : name) (simple : simple) :
      Keyed.lookup name_eqb n (Join_env.equations_of extension)
      = Some (Type_naked_immediate (Ok (Equals simple))) ->
      name_of_simple simple = Some n' ->
      alias_chain extension n' n'' ->
      alias_chain extension n n''.

Definition added_head (extension : env_extension) (n : name)
    (head : head_of_kind_naked_immediate) : Prop :=
  head = Join_env.empty_naked_immediate_head
  ∨ Keyed.lookup name_eqb n (Join_env.equations_of extension)
    = Some (Type_naked_immediate (Ok (No_alias head))).

Ltac added_here n head :=
  right; exists n, head; split; [ apply Chain_here | ].

Lemma add_inverse_relation_spec :
  ∀ (fuel : nat) (seen : list name) (extension : env_extension) (n : name)
    (relation : Relation.t) (scrutinee : name) (extension' : env_extension),
    Join_env.add_inverse_relation_to_env_extension fuel seen extension n
      relation scrutinee
    = Some extension' ->
    extension' = extension
    ∨ ∃ (n' : name) (head : head_of_kind_naked_immediate),
        alias_chain extension n n'
        ∧ added_head extension n' head
        ∧ extension'
          = Join_env.add_or_replace_equation extension n'
              (Join_env.with_inverse_relation head relation scrutinee).
Proof.
  induction fuel as [ | fuel IH];
    intros seen extension n relation scrutinee extension' Hadd;
    cbn [Join_env.add_inverse_relation_to_env_extension] in Hadd.
  - destruct (Keyed.lookup name_eqb n (Join_env.equations_of extension))
      as [existing | ] eqn:Hlookup.
    2:{ injection Hadd as <-.
        added_here n Join_env.empty_naked_immediate_head.
        split; [ left; reflexivity | reflexivity ]. }
    destruct existing as [d | d | k d | d | d]; cbv beta iota in Hadd;
      try discriminate.
    destruct d as [ | [head | simple] | ]; cbv beta iota in Hadd.
    + injection Hadd as <-.
      added_here n Join_env.empty_naked_immediate_head.
      split; [ left; reflexivity | reflexivity ].
    + destruct (naked_immediates head) as [[ | number numbers] | ];
        [ discriminate | | ]; injection Hadd as <-; added_here n head;
        (split; [ right; exact Hlookup | reflexivity ]).
    + destruct (name_of_simple simple) as [name' | ];
        injection Hadd as <-; [ | left; reflexivity ].
      added_here n Join_env.empty_naked_immediate_head.
      split; [ left; reflexivity | reflexivity ].
    + injection Hadd as <-. left. reflexivity.
  - destruct (Keyed.lookup name_eqb n (Join_env.equations_of extension))
      as [existing | ] eqn:Hlookup.
    2:{ injection Hadd as <-.
        added_here n Join_env.empty_naked_immediate_head.
        split; [ left; reflexivity | reflexivity ]. }
    destruct existing as [d | d | k d | d | d]; cbv beta iota in Hadd;
      try discriminate.
    destruct d as [ | [head | simple] | ]; cbv beta iota in Hadd.
    + injection Hadd as <-.
      added_here n Join_env.empty_naked_immediate_head.
      split; [ left; reflexivity | reflexivity ].
    + destruct (naked_immediates head) as [[ | number numbers] | ];
        [ discriminate | | ]; injection Hadd as <-; added_here n head;
        (split; [ right; exact Hlookup | reflexivity ]).
    + destruct (name_of_simple simple) as [name' | ] eqn:Hname;
        [ | injection Hadd as <-; left; reflexivity ].
      destruct (Keyed.among name_eqb name' seen).
      * injection Hadd as <-.
        added_here n Join_env.empty_naked_immediate_head.
        split; [ left; reflexivity | reflexivity ].
      * destruct (IH _ _ _ _ _ _ Hadd)
          as [-> | [n' [head [Hchain [Hhead ->]]]]]; [ left; reflexivity | ].
        right. exists n', head.
        split; [ exact (Chain_step _ _ _ _ _ Hlookup Hname Hchain) | ].
        split; [ exact Hhead | reflexivity ].
    + injection Hadd as <-. left. reflexivity.
Qed.

Lemma name_of_simple_some (simple : simple) (n : name) :
  name_of_simple simple = Some n -> simple = simple_of_name n.
Proof.
  destruct simple as [x | sym | c]; cbn; intros H;
    [ injection H as <- | injection H as <- | discriminate ]; reflexivity.
Qed.

Lemma alias_chain_value (statics : static_env) (vars : env)
    (extension : env_extension) (n n' : name) (w : value) :
  env_extension_satisfied statics vars extension ->
  alias_chain extension n n' ->
  value_of_name vars n = Some w ->
  value_of_name vars n' = Some w.
Proof.
  intros Hsatisfied Hchain.
  induction Hchain as [n | n n' n'' simple Hlookup Hname Hchain IH];
    intros Hw; [ exact Hw | ].
  apply IH. destruct extension as [equations].
  destruct (proj1 (env_extension_lookup statics vars equations) Hsatisfied n _
              Hlookup) as [v [Hv Hty]].
  rewrite Hw in Hv. injection Hv as <-.
  destruct Hty as [Heval _].
  rewrite (name_of_simple_some _ _ Hname) in Heval. exact Heval.
Qed.

Lemma alias_chain_free (extension : env_extension) (n n' : name)
    (y : variable) :
  alias_chain extension n n' ->
  y ∈ free_variables_of_name n' ->
  y ∈ free_variables_of_name n
  ∨ y ∈ free_variables_of_env_extension extension.
Proof.
  intros Hchain.
  induction Hchain as [n | n n' n'' simple Hlookup Hname Hchain IH];
    intros Hy; [ left; exact Hy | ].
  right. destruct (IH Hy) as [Hn' | Hext]; [ | exact Hext ].
  destruct extension as [equations].
  apply (keyed_lookup_in name_eqb name_eqb_decides_equality) in Hlookup.
  cbn [free_variables_of_env_extension]. apply in_flat_map.
  exists (n, Type_naked_immediate (Ok (Equals simple))).
  split; [ exact Hlookup | ]. apply in_or_app. right.
  rewrite (name_of_simple_some _ _ Hname). exact Hn'.
Qed.

Lemma add_inverse_relation_satisfied (statics : static_env) (vars : env)
    (fuel : nat) (seen : list name) (extension : env_extension) (n : name)
    (relation : Relation.t) (scrutinee : name) (extension' : env_extension)
    (i : Z) (sv : value) :
  Join_env.add_inverse_relation_to_env_extension fuel seen extension n
    relation scrutinee
  = Some extension' ->
  env_extension_satisfied statics vars extension ->
  value_of_name vars n = Some (Value_const (Naked_immediate i)) ->
  value_of_name vars scrutinee = Some sv ->
  relation_holds statics relation sv i ->
  env_extension_satisfied statics vars extension'.
Proof.
  intros Hadd Hsatisfied Hn Hscrutinee Hrelation.
  destruct (add_inverse_relation_spec _ _ _ _ _ _ _ Hadd)
    as [-> | [n' [head [Hchain [Hhead ->]]]]]; [ exact Hsatisfied | ].
  pose proof (alias_chain_value statics vars extension n n' _ Hsatisfied
                Hchain Hn) as Hn'.
  destruct extension as [equations].
  pose proof (proj1 (env_extension_lookup statics vars equations) Hsatisfied)
    as Hold.
  cbn [Join_env.add_or_replace_equation Join_env.equations_of] in *.
  apply env_extension_lookup. intros m ty Hm.
  rewrite keyed_lookup_cons in Hm.
  destruct (name_eqb m n') eqn:Heq.
  2:{ rewrite (keyed_lookup_unbind name name_eqb name_eqb_decides_equality),
        Heq in Hm.
      exact (Hold m ty Hm). }
  apply name_eqb_decides_equality in Heq. subst m. injection Hm as <-.
  exists (Value_const (Naked_immediate i)). split; [ exact Hn' | ].
  assert (Hhead' :
            head = Join_env.empty_naked_immediate_head
            ∨ head_of_kind_naked_immediate_satisfied statics vars
                (Value_const (Naked_immediate i)) head).
  { destruct Hhead as [-> | Hhead]; [ left; reflexivity | right ].
    destruct (Hold n' _ Hhead) as [v' [Hv' Hty']].
    rewrite Hn' in Hv'. injection Hv' as <-. exact Hty'. }
  unfold Join_env.with_inverse_relation.
  change
    (head_of_kind_naked_immediate_satisfied statics vars
       (Value_const (Naked_immediate i))
       (Head_of_kind_naked_immediate (naked_immediates head)
          ((relation, scrutinee) :: inverse_relations head))).
  exists i. split; [ reflexivity | ].
  cbn [naked_immediates inverse_relations]. split.
  - destruct Hhead' as [-> | [i' [Hi' [Hnumbers _]]]]; [ exact I | ].
    injection Hi' as <-. exact Hnumbers.
  - intros rel m [Heq' | Hin].
    + injection Heq' as <- <-. exists sv.
      split; [ exact Hscrutinee | exact Hrelation ].
    + destruct Hhead' as [-> | [i' [Hi' [_ Hrelations]]]]; [ destruct Hin | ].
      injection Hi' as <-. exact (Hrelations rel m Hin).
Qed.

Lemma add_inverse_relation_free_variables (fuel : nat) (seen : list name)
    (extension : env_extension) (n : name) (relation : Relation.t)
    (scrutinee : name) (extension' : env_extension) (y : variable) :
  Join_env.add_inverse_relation_to_env_extension fuel seen extension n
    relation scrutinee
  = Some extension' ->
  y ∈ free_variables_of_env_extension extension' ->
  y ∈ free_variables_of_env_extension extension
  ∨ y ∈ free_variables_of_name n
  ∨ y ∈ free_variables_of_name scrutinee.
Proof.
  intros Hadd Hy.
  destruct (add_inverse_relation_spec _ _ _ _ _ _ _ Hadd)
    as [-> | [n' [head [Hchain [Hhead ->]]]]]; [ left; exact Hy | ].
  assert (Hcases :
            y ∈ free_variables_of_name n'
            ∨ y ∈ free_variables_of_name scrutinee
            ∨ y ∈ free_variables_of_naked_immediate head
            ∨ y ∈ free_variables_of_env_extension extension).
  { destruct extension as [equations].
    cbn [Join_env.add_or_replace_equation free_variables_of_env_extension
           flat_map fst snd] in Hy.
    apply in_app_or in Hy. destruct Hy as [Hy | Hy].
    2:{ right. right. right. apply in_flat_map in Hy.
        destruct Hy as [binding [Hbinding Hy]]. apply filter_In in Hbinding.
        apply in_flat_map. exists binding.
        split; [ exact (proj1 Hbinding) | exact Hy ]. }
    apply in_app_or in Hy. destruct Hy as [Hy | Hy]; [ left; exact Hy | ].
    right. unfold Join_env.with_inverse_relation in Hy.
    cbn [free_variables free_variables_of_descr
           free_variables_of_naked_immediate inverse_relations flat_map snd]
      in Hy.
    apply in_app_or in Hy. destruct Hy as [Hy | Hy];
      [ left; exact Hy | right; left; exact Hy ]. }
  destruct Hcases as [Hy' | [Hy' | [Hy' | Hy']]].
  - destruct (alias_chain_free extension n n' y Hchain Hy') as [H | H];
      [ right; left; exact H | left; exact H ].
  - right. right. exact Hy'.
  - destruct Hhead as [-> | Hhead]; [ destruct Hy' | ].
    left. destruct extension as [equations].
    apply (keyed_lookup_in name_eqb name_eqb_decides_equality) in Hhead.
    cbn [free_variables_of_env_extension]. apply in_flat_map.
    exists (n', Type_naked_immediate (Ok (No_alias head))).
    split; [ exact Hhead | ]. apply in_or_app. right. exact Hy'.
  - left. exact Hy'.
Qed.

Definition variant_type (is_int : option variable)
    (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) : flambda_type :=
  Type_value
    (Ok
       (No_alias
          (Head_of_kind_value
             (Ok (Variant is_int immediates get_tag blocks extensions
                    is_unique))
             Not_null))).

Definition extensions_pair (extensions : variant_extensions)
    : env_extension * env_extension :=
  match extensions with
  | No_extensions => (empty_env_extension, empty_env_extension)
  | Ext when_immediate when_block => (when_immediate, when_block)
  end.

(* The type that [recover_inverse_relations] answers is the type given,
   or a type of variants with the relation [Get_tag] added in its
   extension for blocks. *)
Lemma recover_type_cases (relations relations' : Join_env.inverse_relations)
    (n : name) (ty ty' : flambda_type) :
  Join_env.recover_inverse_relations relations n ty = Some (ty', relations')
  ->
  ty' = ty
  ∨ ∃ (is_int : option variable) (immediates : Or_unknown.t flambda_type)
      (g : variable) (blocks : Or_unknown.t row_like_for_blocks)
      (extensions : variant_extensions) (is_unique : bool)
      (when_block : env_extension),
      ty = variant_type is_int immediates (Some g) blocks extensions is_unique
      ∧ Join_env.add_inverse_relation_to_env_extension
          (length (Join_env.equations_of (snd (extensions_pair extensions))))
          [] (snd (extensions_pair extensions)) (Name_var g) Relation.Get_tag
          n
        = Some when_block
      ∧ ty'
        = variant_type is_int immediates (Some g) blocks
            (Ext (fst (extensions_pair extensions)) when_block) is_unique.
Proof.
  intros Hrecover. unfold Join_env.recover_inverse_relations in Hrecover.
  destruct ty as [d | d | k d | d | d];
    try (injection Hrecover as <- _; left; reflexivity).
  destruct d as [ | [head | simple] | ];
    try (injection Hrecover as <- _; left; reflexivity).
  destruct head as [non_null is_null].
  destruct is_null as [ | is_null].
  2:{ destruct non_null as [ | head | ]; injection Hrecover as <- _; left;
        reflexivity. }
  destruct non_null as [ | head | ];
    try (injection Hrecover as <- _; left; reflexivity).
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique | | | | ];
    try (injection Hrecover as <- _; left; reflexivity).
  cbv zeta in Hrecover.
  destruct get_tag as [g | ].
  2:{ match type of Hrecover with
      | context [match ?e with true => _ | false => _ end] => destruct e
      end; injection Hrecover as <- _; left; reflexivity. }
  match type of Hrecover with
  | context [match ?e with true => _ | false => _ end] => destruct e
  end.
  { injection Hrecover as <- _. left. reflexivity. }
  destruct extensions as [ | when_immediate when_block];
    cbv beta iota zeta in Hrecover;
    (destruct (Join_env.add_inverse_relation_to_env_extension _ _ _ _ _ _)
       as [when_block' | ] eqn:Hadd; [ | discriminate ]);
    injection Hrecover as <- _; right.
  - exists is_int, immediates, g, blocks, No_extensions, is_unique,
      when_block'.
    split; [ reflexivity | split; [ exact Hadd | reflexivity ] ].
  - exists is_int, immediates, g, blocks, (Ext when_immediate when_block),
      is_unique, when_block'.
    split; [ reflexivity | split; [ exact Hadd | reflexivity ] ].
Qed.

(* The relations that it answers are those given, or those with one
   more: [Get_tag] for the variable of the tag of a type of variants
   that has no immediates, [Is_int] for the variable of a type of
   variants, [Is_null] for the variable of a type that may be null. *)
Lemma recover_relations_cases
    (relations relations' : Join_env.inverse_relations) (n : name)
    (ty ty' : flambda_type) :
  Join_env.recover_inverse_relations relations n ty = Some (ty', relations')
  ->
  relations' = relations
  ∨ (∃ (is_int : option variable) (imm_ty : flambda_type) (g : variable)
       (blocks : Or_unknown.t row_like_for_blocks)
       (extensions : variant_extensions) (is_unique : bool),
       ty
       = variant_type is_int (Or_unknown.Known imm_ty) (Some g) blocks
           extensions is_unique
       ∧ is_obviously_bottom imm_ty = true
       ∧ relations'
         = Join_env.add_to_inverse_relations relations g Relation.Get_tag n)
  ∨ (∃ (r : variable) (immediates : Or_unknown.t flambda_type)
       (get_tag : option variable)
       (blocks : Or_unknown.t row_like_for_blocks)
       (extensions : variant_extensions) (is_unique : bool),
       ty
       = variant_type (Some r) immediates get_tag blocks extensions is_unique
       ∧ relations'
         = Join_env.add_to_inverse_relations relations r Relation.Is_int n)
  ∨ ∃ (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
      (r : variable),
      ty
      = Type_value
          (Ok (No_alias (Head_of_kind_value non_null (Maybe_null (Some r)))))
      ∧ relations'
        = Join_env.add_to_inverse_relations relations r Relation.Is_null n.
Proof.
  intros Hrecover. unfold Join_env.recover_inverse_relations in Hrecover.
  destruct ty as [d | d | k d | d | d];
    try (injection Hrecover as _ <-; left; reflexivity).
  destruct d as [ | [head | simple] | ];
    try (injection Hrecover as _ <-; left; reflexivity).
  destruct head as [non_null is_null].
  destruct is_null as [ | [r | ]].
  2:{ right. right. right. exists non_null, r.
      destruct non_null as [ | head | ]; injection Hrecover as _ <-;
        split; reflexivity. }
  2:{ destruct non_null as [ | head | ]; injection Hrecover as _ <-; left;
        reflexivity. }
  destruct non_null as [ | head | ];
    try (injection Hrecover as _ <-; left; reflexivity).
  destruct head
    as [is_int immediates get_tag blocks extensions is_unique | | | | ];
    try (injection Hrecover as _ <-; left; reflexivity).
  cbv zeta in Hrecover.
  destruct get_tag as [g | ].
  2:{ match type of Hrecover with
      | context [match ?e with true => _ | false => _ end] => destruct e
      end;
        (destruct is_int as [r | ]; injection Hrecover as _ <-;
         [ right; right; left; do 6 eexists; split; reflexivity
         | left; reflexivity ]). }
  match type of Hrecover with
  | context [match ?e with true => _ | false => _ end] =>
    destruct e eqn:Hno
  end.
  { injection Hrecover as _ <-. right. left.
    destruct immediates as [imm_ty | ]; [ | discriminate ].
    exists is_int, imm_ty, g, blocks, extensions, is_unique.
    split; [ reflexivity | split; [ exact Hno | reflexivity ] ]. }
  destruct extensions as [ | when_immediate when_block];
    cbv beta iota zeta in Hrecover;
    (destruct (Join_env.add_inverse_relation_to_env_extension _ _ _ _ _ _)
       as [when_block' | ]; [ | discriminate ]);
    (destruct is_int as [r | ]; injection Hrecover as _ <-;
     [ right; right; left; do 6 eexists; split; reflexivity
     | left; reflexivity ]).
Qed.

Lemma variant_free_variables (is_int : option variable)
    (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) (y : variable) :
  y
  ∈ free_variables
      (variant_type is_int immediates get_tag blocks extensions is_unique)
  <-> y ∈ free_variables_of_option is_int
      ∨ y
        ∈ match immediates with
          | Or_unknown.Known ty => free_variables ty
          | Or_unknown.Unknown => []
          end
      ∨ y ∈ free_variables_of_option get_tag
      ∨ y
        ∈ match blocks with
          | Or_unknown.Known blocks =>
            free_variables_of_row_like_for_blocks blocks
          | Or_unknown.Unknown => []
          end
      ∨ y
        ∈ free_variables_of_env_extension (fst (extensions_pair extensions))
      ∨ y
        ∈ free_variables_of_env_extension (snd (extensions_pair extensions)).
Proof.
  unfold variant_type.
  cbn [free_variables free_variables_of_descr
         free_variables_of_head_of_kind_value
         free_variables_of_head_of_kind_value_non_null].
  rewrite !in_app_iff.
  destruct extensions as [ | when_immediate when_block];
    cbn [extensions_pair fst snd empty_env_extension
           free_variables_of_env_extension flat_map In];
    [ | rewrite in_app_iff ]; tauto.
Qed.

Theorem recover_free_variables : recover_free_variables_statement.
Proof.
  intros relations relations' x ty ty' y Hrecover Hy.
  destruct (recover_type_cases _ _ _ _ _ Hrecover)
    as [-> | [is_int [immediates [g [blocks [extensions [is_unique
         [when_block [-> [Hadd ->]]]]]]]]]]; [ left; exact Hy | ].
  rewrite variant_free_variables in Hy. rewrite variant_free_variables.
  cbn [extensions_pair fst snd] in Hy.
  destruct Hy as [Hy | [Hy | [Hy | [Hy | [Hy | Hy]]]]]; try tauto.
  destruct (add_inverse_relation_free_variables _ _ _ _ _ _ _ y Hadd Hy)
    as [H | [[<- | []] | [<- | []]]].
  - tauto.
  - left. right. right. left. left. reflexivity.
  - right. reflexivity.
Qed.

Lemma recorded_add_cases (relations : Join_env.inverse_relations)
    (var result : variable) (relation relation' : Relation.t)
    (scrutinee scrutinee' : name) :
  relation_recorded
    (Join_env.add_to_inverse_relations relations var relation' scrutinee')
    result relation scrutinee ->
  relation_recorded relations result relation scrutinee
  ∨ (result = var ∧ relation = relation' ∧ scrutinee = scrutinee').
Proof.
  intros [recorded [Hlookup Hin]].
  unfold Join_env.add_to_inverse_relations in Hlookup.
  cbn [lookup] in Hlookup.
  destruct (Pos.eqb result var) eqn:Heq.
  - apply Pos.eqb_eq in Heq. subst var. injection Hlookup as <-.
    destruct Hin as [Hhead | Hin].
    + injection Hhead as <- <-. right. repeat split.
    + destruct (lookup result relations) as [existing | ] eqn:Hexisting.
      * left. exists existing. split; [ exact Hexisting | exact Hin ].
      * destruct Hin.
  - left. exists recorded. split; assumption.
Qed.

Lemma variant_satisfied (statics : static_env) (vars : env) (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  type_satisfied statics vars v
    (variant_type is_int immediates get_tag blocks extensions is_unique)
  <-> v <> Value_const Null
      ∧ head_of_kind_value_non_null_satisfied statics vars v
          (Variant is_int immediates get_tag blocks extensions is_unique).
Proof.
  unfold variant_type.
  change
    (type_satisfied statics vars v
       (Type_value
          (Ok
             (No_alias
                (Head_of_kind_value
                   (Ok
                      (Variant is_int immediates get_tag blocks extensions
                         is_unique))
                   Not_null)))))
    with
    ((v = Value_const Null ∧ False)
     ∨ (v <> Value_const Null
        ∧ head_of_kind_value_non_null_satisfied statics vars v
            (Variant is_int immediates get_tag blocks extensions is_unique)
        ∧ True)).
  tauto.
Qed.

Theorem recovered_relations_hold : recovered_relations_hold_statement.
Proof.
  intros relations relations' x ty ty' r relation n Hrecover Hrecorded.
  destruct (recover_relations_cases _ _ _ _ _ Hrecover)
    as [-> | [Hcase | [Hcase | Hcase]]]; [ left; exact Hrecorded | | | ].
  - destruct Hcase
      as [is_int [imm_ty [g [blocks [extensions [is_unique [-> [Hno ->]]]]]]]].
    destruct (recorded_add_cases _ _ _ _ _ _ _ Hrecorded)
      as [Hold | [-> [-> ->]]]; [ left; exact Hold | right ].
    split; [ reflexivity | ]. split.
    + apply variant_free_variables. right. right. left. left. reflexivity.
    + intros statics vars v Htags Hty. apply variant_satisfied in Hty.
      destruct Hty as [_ Hty].
      destruct Hty
        as [[i [_ [_ [Himm _]]]]
           | [tag [fields [Hblock [_ [Htag _]]]]]].
      * exfalso.
        exact (is_obviously_bottom_satisfied statics vars _ imm_ty Hno Himm).
      * exists tag. split; [ exact Htag | ]. right.
        split; [ exists fields; exact Hblock | ].
        exact (Htags tag fields Hblock).
  - destruct Hcase
      as [r0 [immediates [get_tag [blocks [extensions [is_unique [-> ->]]]]]]].
    destruct (recorded_add_cases _ _ _ _ _ _ _ Hrecorded)
      as [Hold | [-> [-> ->]]]; [ left; exact Hold | right ].
    split; [ reflexivity | ]. split.
    + apply variant_free_variables. left. left. reflexivity.
    + intros statics vars v _ Hty. apply variant_satisfied in Hty.
      destruct Hty as [_ Hty].
      exact (is_int_coherent statics vars v r0 immediates get_tag blocks
               extensions is_unique Hty).
  - destruct Hcase as [non_null [r0 [-> ->]]].
    destruct (recorded_add_cases _ _ _ _ _ _ _ Hrecorded)
      as [Hold | [-> [-> ->]]]; [ left; exact Hold | right ].
    split; [ reflexivity | ]. split.
    + cbn [free_variables free_variables_of_descr
             free_variables_of_head_of_kind_value free_variables_of_option].
      apply in_or_app. right. left. reflexivity.
    + intros statics vars v _ Hty.
      exact (is_null_coherent statics vars v non_null r0 Hty).
Qed.

Theorem recover_keeps_meaning : recover_keeps_meaning_statement.
Proof.
  intros relations relations' x ty ty' statics vars v Hrecover Hx Htags Hty.
  destruct (recover_type_cases _ _ _ _ _ Hrecover)
    as [-> | [is_int [immediates [g [blocks [extensions [is_unique
         [when_block [-> [Hadd ->]]]]]]]]]]; [ exact Hty | ].
  apply variant_satisfied in Hty. apply variant_satisfied.
  destruct Hty as [Hnot_null Hty]. split; [ exact Hnot_null | ].
  destruct Hty
    as [[i [Hv [Hint [Himm Hext]]]]
       | [tag [fields [Hblock [Hint [Htag [Hblocks Hext]]]]]]].
  - left. exists i. split; [ exact Hv | ]. split; [ exact Hint | ].
    split; [ exact Himm | ].
    destruct extensions as [ | when_immediate when_block0];
      [ | exact Hext ].
    apply env_extension_lookup. intros m t Hm. discriminate.
  - right. exists tag, fields. split; [ exact Hblock | ].
    split; [ exact Hint | ]. split; [ exact Htag | ].
    split; [ exact Hblocks | ].
    apply (add_inverse_relation_satisfied statics vars _ _ _ _ _ _ _ tag v
             Hadd); [ | exact Htag | exact Hx | ].
    + destruct extensions as [ | when_immediate when_block0];
        [ | exact Hext ].
      apply env_extension_lookup. intros m t Hm. discriminate.
    + right. split; [ exists fields; exact Hblock | ].
      exact (Htags tag fields Hblock).
Qed.

Lemma later_relations_hold (s s' : Join_env.t)
    (relations : Join_env.inverse_relations) :
  later s s' -> relations_hold s relations -> relations_hold s' relations.
Proof.
  intros Hlater Hrelations r relation n Hrecorded.
  destruct (Hrelations r relation n Hrecorded)
    as [x [Hn [Hx [Hr Hholds]]]].
  exists x. split; [ exact Hn | ].
  split; [ exact (later_exists _ _ _ Hlater Hx) | ].
  split; [ exact (later_exists _ _ _ Hlater Hr) | ].
  intros index use statics valuation v Huse Hsatisfied Htags Hv.
  rewrite (later_env _ _ _ Hlater) in Huse.
  rewrite (later_target_valuation s s' index use valuation x Hlater
             (or_introl Hx)) in Hv.
  rewrite (later_target_valuation s s' index use valuation r Hlater
             (or_introl Hr)).
  exact (Hholds _ _ _ _ _ Huse Hsatisfied
           (later_tags_at _ _ _ _ _ Hlater Htags) Hv).
Qed.

(* ---------------------------------------------------------------------- *)
(* The rounds                                                              *)
(* ---------------------------------------------------------------------- *)

Lemma insert_keys {A : Type} (b0 : Index.t * A) (sorted : list (Index.t * A))
    (k : Index.t) :
  In k (map fst (Index.insert b0 sorted))
  <-> k = fst b0 ∨ In k (map fst sorted).
Proof.
  rewrite !in_map_iff. split.
  - intros [b [Hk Hin]]. apply in_insert in Hin. destruct Hin as [-> | Hin].
    + left. symmetry. exact Hk.
    + right. exists b. split; [ exact Hk | exact Hin ].
  - intros [-> | [b [Hk Hin]]].
    + exists b0. split; [ reflexivity | ]. apply in_insert. left. reflexivity.
    + exists b. split; [ exact Hk | ]. apply in_insert. right. exact Hin.
Qed.

Lemma insert_distinct {A : Type} (b0 : Index.t * A) :
  ∀ sorted : list (Index.t * A),
    NoDup (map fst sorted) ->
    ~ In (fst b0) (map fst sorted) ->
    NoDup (map fst (Index.insert b0 sorted)).
Proof.
  induction sorted as [ | b' rest IH]; intros Hdistinct Hfresh;
    cbn [Index.insert].
  - cbn. constructor; [ intros [] | constructor ].
  - destruct (Nat.leb (fst b0) (fst b')).
    + cbn [map]. constructor; [ exact Hfresh | exact Hdistinct ].
    + cbn [map] in *. inversion Hdistinct as [ | ? ? Hb' Hrest]; subst.
      constructor.
      * rewrite insert_keys. intros [Heq | Hin]; [ | exact (Hb' Hin) ].
        apply Hfresh. left. exact Heq.
      * apply IH; [ exact Hrest | ]. intros Hin. apply Hfresh. right.
        exact Hin.
Qed.

Lemma bindings_distinct {A : Type} (m : Keyed.finite_map Index.t A) :
  NoDup (map fst (Index.bindings m)).
Proof.
  unfold Index.bindings.
  pose proof (proj1 (keyed_distinct_NoDup Index.equal index_equal_decides _)
                (keyed_compact_distinct Index.t Index.equal
                   index_equal_decides _ m)) as Hdistinct.
  revert Hdistinct. generalize (Keyed.compact Index.equal m) as l.
  induction l as [ | b l IH]; intros Hdistinct; cbn [List.fold_right].
  - constructor.
  - cbn [map] in Hdistinct. inversion Hdistinct as [ | ? ? Hb Hrest]; subst.
    apply insert_distinct; [ exact (IH Hrest) | ].
    intros Hin. apply Hb. clear -Hin.
    induction l as [ | b' l IH]; [ destruct Hin | ].
    cbn [List.fold_right] in Hin. apply insert_keys in Hin.
    destruct Hin as [Heq | Hin]; [ left; symmetry; exact Heq | ].
    right. exact (IH Hin).
Qed.

(* The value that a variable has in the valuation of the target is the
   value of a [Simple] of variables that the input or the source
   knows. *)
Lemma target_value_of_simple (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) (x : variable) (v : value) :
  state_holds s ->
  Join_env.joined_env s index = Some use ->
  lookup x (target_valuation s index use valuation) = Some v ->
  ∃ simple : simple,
    (∀ y : variable,
       y ∈ free_variables_of_simple simple -> known_to s use y = true)
    ∧ eval_simple valuation simple = Some v.
Proof.
  intros [_ [_ [_ Hvectors]]] Huse Hv. rewrite lookup_target_valuation in Hv.
  destruct (value_of_definition s index valuation x) as [v' | ] eqn:Hdefined.
  - injection Hv as <-. unfold value_of_definition in Hdefined.
    destruct (lookup x (Join_env.definitions_of_existentials s))
      as [[simples kind] | ] eqn:Hdefinition; [ | discriminate ].
    destruct (Simples_in_joined_envs.find index simples) as [simple | ]
      eqn:Hfind; [ | discriminate ].
    exists simple. split; [ | exact Hdefined ].
    destruct (Hvectors x simples kind Hdefinition index simple Hfind)
      as [use' [Huse' Hscope]].
    pose proof (eq_trans (eq_sym Huse') Huse) as Heq. injection Heq as ->.
    intros y Hy. unfold known_to. rewrite (Hscope y Hy). reflexivity.
  - destruct (known_to s use x) eqn:Hknown; [ | discriminate ].
    exists (Var x). split; [ | exact Hv ].
    intros y [<- | []]. exact Hknown.
Qed.

(* The type that a joined environment has for a [Simple]. *)
Definition type_of_simple_at (expand_const : const -> flambda_type)
    (envs : Joined_envs.t) (kind : flambda_kind) (index : Index.t)
    (simple : simple) : option flambda_type :=
  match simple with
  | Const const => Some (expand_const const)
  | Var var =>
    let* env := Joined_envs.get_nth_joined_env envs index in
    Typing_env.find env (Name_var var) (Some kind)
  | Symbol symbol =>
    let* env := Joined_envs.get_nth_joined_env envs index in
    Typing_env.find env (Name_symbol symbol) (Some kind)
  end.

Lemma types_of_simples_cons (expand_const : const -> flambda_type)
    (envs : Joined_envs.t) (kind : flambda_kind) (index : Index.t)
    (simple : simple) (simples : list (Index.t * FlambdaModel.simple)) :
  Joined_envs.types_of_simples expand_const envs kind
    ((index, simple) :: simples)
  = let* ty := type_of_simple_at expand_const envs kind index simple in
    let* types :=
      Joined_envs.types_of_simples expand_const envs kind simples
    in
    Some ((index, ty) :: types).
Proof. destruct simple; reflexivity. Qed.

Lemma types_of_simples_lookup (expand_const : const -> flambda_type)
    (envs : Joined_envs.t) (kind : flambda_kind) :
  ∀ (simples : list (Index.t * simple))
    (types : Keyed.finite_map Index.t flambda_type),
    Joined_envs.types_of_simples expand_const envs kind simples = Some types
    ->
    (∀ (index : Index.t) (t : flambda_type),
       Keyed.lookup Index.equal index types = Some t ->
       ∃ simple : simple,
         Keyed.lookup Index.equal index simples = Some simple
         ∧ type_of_simple_at expand_const envs kind index simple = Some t)
    ∧ ∀ (index : Index.t) (simple : simple),
        Keyed.lookup Index.equal index simples = Some simple ->
        ∃ t : flambda_type,
          Keyed.lookup Index.equal index types = Some t
          ∧ type_of_simple_at expand_const envs kind index simple = Some t.
Proof.
  induction simples as [ | [index0 simple0] simples IH]; intros types H.
  - injection H as <-. split; intros index x Hx; discriminate.
  - rewrite types_of_simples_cons in H.
    destruct (type_of_simple_at expand_const envs kind index0 simple0)
      as [ty0 | ] eqn:Hty0; [ | discriminate ].
    destruct (Joined_envs.types_of_simples expand_const envs kind simples)
      as [types0 | ]; [ | discriminate ].
    injection H as <-. destruct (IH _ eq_refl) as [Hfound Hcovered]. split.
    + intros index t Ht. rewrite keyed_lookup_cons in Ht.
      rewrite keyed_lookup_cons.
      destruct (Index.equal index index0) eqn:Heq.
      * injection Ht as <-. apply index_equal_decides in Heq. subst index0.
        exists simple0. split; [ reflexivity | exact Hty0 ].
      * exact (Hfound index t Ht).
    + intros index simple Hsimple. rewrite keyed_lookup_cons in Hsimple.
      rewrite keyed_lookup_cons.
      destruct (Index.equal index index0) eqn:Heq.
      * injection Hsimple as <-. apply index_equal_decides in Heq.
        subst index0. exists ty0. split; [ reflexivity | exact Hty0 ].
      * exact (Hcovered index simple Hsimple).
Qed.

Lemma equations_of_existentials_lookup (p : Join_env.parameters)
    (s : Join_env.t) :
  ∀ (existentials :
       list (variable * (Simples_in_joined_envs.t * flambda_kind)))
    (equations : finite_map (Keyed.finite_map Index.t flambda_type)),
    Join_env.equations_of_existentials p s existentials = Some equations ->
    ∀ (x : variable) (types : Keyed.finite_map Index.t flambda_type),
      lookup x equations = Some types ->
      ∃ (simples : Simples_in_joined_envs.t) (kind : flambda_kind),
        In (x, (simples, kind)) existentials
        ∧ Joined_envs.find_simples_in_joined_envs (Join_env.expand_const p)
            (Join_env.joined_envs s) simples kind
          = Some types.
Proof.
  induction existentials as [ | [x0 [simples0 kind0]] rest IH];
    intros equations H x types Hx.
  - injection H as <-. discriminate.
  - cbn [Join_env.equations_of_existentials] in H.
    destruct (Joined_envs.find_simples_in_joined_envs
                (Join_env.expand_const p) (Join_env.joined_envs s) simples0
                kind0) as [types0 | ] eqn:Hfind; [ | discriminate ].
    destruct (Join_env.equations_of_existentials p s rest)
      as [equations0 | ]; [ | discriminate ].
    injection H as <-. rewrite lookup_cons in Hx.
    destruct (Pos.eqb x x0) eqn:Heq.
    + injection Hx as <-. apply Pos.eqb_eq in Heq. subst x0.
      exists simples0, kind0. split; [ left; reflexivity | exact Hfind ].
    + destruct (IH _ eq_refl _ _ Hx) as [simples [kind [Hin Hf]]].
      exists simples, kind. split; [ right; exact Hin | exact Hf ].
Qed.

(* A variable that the join has made has, where it has a value in the
   valuation of the target, the value of the [Simple] of its
   definition. *)
Lemma existential_value (s : Join_env.t) (x : variable)
    (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
    (index : Index.t) (use : Typing_env.t) (valuation : env) (v : value) :
  state_valid s ->
  lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind) ->
  Join_env.joined_env s index = Some use ->
  lookup x (target_valuation s index use valuation) = Some v ->
  ∃ simple : simple,
    Simples_in_joined_envs.find index simples = Some simple
    ∧ eval_simple valuation simple = Some v.
Proof.
  intros Hvalid Hdefinition Huse Hv.
  assert (Hunknown : known_to s use x = false).
  { destruct (known_to s use x) eqn:Hknown; [ | reflexivity ].
    pose proof (known_no_definition s index index use valuation x Hvalid Huse
                  Hknown) as Hnone.
    exfalso. destruct Hvalid as [_ [Hnew [[Hrecorded _] _]]].
    assert (Hx : is_existential s x).
    { unfold is_existential. apply is_bound_lookup. exists kind.
      exact (Hrecorded _ _ _ Hdefinition). }
    destruct (Hnew x Hx) as [Hsource Huses].
    unfold known_to in Hknown. apply orb_true_iff in Hknown.
    destruct Hknown as [Hk | Hk];
      [ exact (Huses _ _ Huse Hk) | exact (Hsource Hk) ]. }
  rewrite lookup_target_valuation, Hunknown in Hv.
  unfold value_of_definition in Hv. rewrite Hdefinition in Hv.
  destruct (Simples_in_joined_envs.find index simples) as [simple | ];
    [ | discriminate ].
  exists simple. split; [ reflexivity | ].
  destruct (eval_simple valuation simple) as [v' | ]; [ | discriminate ].
  exact Hv.
Qed.

(* The types that the joined environments have for the [Simple]s of
   the definition of a variable are true of it. *)
Lemma existential_types_hold (p : Join_env.parameters) (s : Join_env.t)
    (x : variable) (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
    (types : Keyed.finite_map Index.t flambda_type) :
  expand_const_sound p ->
  state_valid s ->
  state_holds s ->
  envs_invariant s ->
  envs_kinded s ->
  lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind) ->
  Joined_envs.find_simples_in_joined_envs (Join_env.expand_const p)
    (Join_env.joined_envs s) simples kind
  = Some types ->
  types_hold s x types.
Proof.
  intros Hconst Hvalid Hholds Hinvariant Hkinded Hdefinition Hfind.
  unfold Joined_envs.find_simples_in_joined_envs in Hfind.
  destruct (types_of_simples_lookup _ _ _ _ _ Hfind) as [Hfound Hcovered].
  pose proof (proj2 (proj2 (proj2 Hholds)) x simples kind Hdefinition)
    as Hvector.
  split.
  { apply existential_in_reach. destruct Hvalid as [_ [_ [[Hrecorded _] _]]].
    unfold is_existential. apply is_bound_lookup. exists kind.
    exact (Hrecorded _ _ _ Hdefinition). }
  split.
  - intros index t Ht. destruct (Hfound index t Ht) as [simple [Hsimple Hty]].
    rewrite (keyed_lookup_compact Index.t Index.equal index_equal_decides)
      in Hsimple.
    destruct (Hvector index simple Hsimple) as [use [Huse Hscope]].
    exists use. split; [ exact Huse | ].
    destruct simple as [y | sym | c]; cbn [type_of_simple_at] in Hty.
    + change (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
        with (Join_env.joined_env s index) in Hty.
      rewrite Huse in Hty. cbn in Hty. split.
      * intros z Hz.
        exact (find_closed use (Name_var y) (Some kind) t z
                 (invariant_types_closed use (Hinvariant _ _ Huse)) Hty Hz).
      * exact (find_kinded_in use _ _ t (Hkinded _ _ Huse) Hty).
    + change (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
        with (Join_env.joined_env s index) in Hty.
      rewrite Huse in Hty. cbn in Hty. split.
      * intros z Hz.
        exact (find_closed use (Name_symbol sym) (Some kind) t z
                 (invariant_types_closed use (Hinvariant _ _ Huse)) Hty Hz).
      * exact (find_kinded_in use _ _ t (Hkinded _ _ Huse) Hty).
    + injection Hty as <-. destruct (Hconst c) as [Hfree [Hkind _]]. split.
      * intros z Hz. rewrite Hfree in Hz. destruct Hz.
      * exact (Hkind use).
  - intros index use statics valuation v Huse Hsatisfied Hv.
    destruct (existential_value s x simples kind index use valuation v Hvalid
                Hdefinition Huse Hv) as [simple [Hsimple Heval]].
    pose proof Hsimple as Hsimple'.
    unfold Simples_in_joined_envs.find in Hsimple'.
    rewrite <- (keyed_lookup_compact Index.t Index.equal index_equal_decides)
      in Hsimple'.
    destruct (Hcovered index simple Hsimple') as [t [Ht Hty]].
    exists t. split; [ exact Ht | ].
    destruct simple as [y | sym | c]; cbn [type_of_simple_at] in Hty.
    + change (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
        with (Join_env.joined_env s index) in Hty.
      rewrite Huse in Hty. cbn in Hty.
      exact (find_satisfied_strictly statics valuation use (Name_var y)
               (Some kind) t v Hsatisfied Hty Heval).
    + change (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
        with (Join_env.joined_env s index) in Hty.
      rewrite Huse in Hty. cbn in Hty.
      exact (find_satisfied_strictly statics valuation use (Name_symbol sym)
               (Some kind) t v Hsatisfied Hty Heval).
    + injection Hty as <-. cbn in Heval. injection Heval as <-.
      exact (proj2 (proj2 (Hconst c)) statics valuation).
Qed.

Section Rounds.

Context (source : Typing_env.t) (jt : Join_env.n_way_join_type)
  (choices : Join_env.choices).

Hypothesis jt_sound :
  FlambdaJoinSpec.n_way_join_type_sound typing_env_invariant
    Join_env.joined_env Join_env.definitions_of_existentials
    kind_in_target_env (state_true source) state_kinded jt.

(* A call of the join of types on the types kept for a variable. *)
Lemma join_call_holds (s s0 : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type)
    (joined : Or_unknown.t flambda_type) :
  state_true source s ->
  envs_invariant s ->
  envs_kinded s ->
  types_hold s var types ->
  jt s (Index.bindings types) = Some (joined, s0) ->
  later s s0
  ∧ state_holds s0
  ∧ ∀ ty : flambda_type,
      joined = Or_unknown.Known ty ->
      type_in_target kind_in_target_env s0 ty
      ∧ ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
          (valuation : env) (v : value),
          Join_env.joined_env s0 index = Some use ->
          typing_env_satisfied_strictly statics valuation use ->
          lookup var (target_valuation s0 index use valuation) = Some v ->
          type_satisfied statics (target_valuation s0 index use valuation) v
            ty.
Proof.
  intros [Hvalid [Hholds Hsource]] Hinvariant Hkinded [Hvar [Hfound Htrue]]
    Hjoin.
  assert (Hscope :
            join_args_in_scope Join_env.joined_env type_in_scope s
              (Index.bindings types)).
  { intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [Hscope _]]].
    exists use. split; [ exact Huse | exact Hscope ]. }
  assert (Hkinds :
            join_args_in_scope Join_env.joined_env type_kinded s
              (Index.bindings types)).
  { intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [_ Hkind]]].
    exists use. split; [ exact Huse | exact Hkind ]. }
  destruct (jt_sound s s0 (Index.bindings types) joined
              (conj Hvalid (conj Hholds Hsource)) Hinvariant Hkinded
              (bindings_distinct types) Hscope Hkinds Hjoin)
    as [[Hvalid0 [Hholds0 Hsource0]] [Hgrows [Hknown _]]].
  assert (Hlater : later s s0).
  { split; [ exact Hvalid | ]. split; [ exact Hvalid0 | ].
    split; [ exact Hgrows | ]. rewrite Hsource0. symmetry. exact Hsource. }
  split; [ exact Hlater | ]. split; [ exact Hholds0 | ].
  intros ty Hty. destruct (Hknown ty Hty) as [Htarget [_ Hupper]].
  split; [ exact Htarget | ].
  intros index use statics valuation v Huse Hsatisfied Hv.
  pose proof Huse as Huse_s. rewrite (later_env _ _ _ Hlater) in Huse_s.
  rewrite (later_target_valuation s s0 index use valuation var Hlater
             (or_introl Hvar)) in Hv.
  destruct (Htrue _ _ _ _ _ Huse_s Hsatisfied Hv) as [t [Ht Hvt]].
  apply (Hupper index t use statics valuation _ v
           (proj2 (bindings_lookup types index t) Ht) Huse Hsatisfied);
    [ | exact Hvt ].
  exact (proj1 (target_valuation_extends_at s0 index use valuation Hvalid0
                  Huse)).
Qed.

(* The equation that a round makes for a variable from the type joined
   for it, with the relations recovered or not. *)
Lemma known_equation_holds (s0 s2 : Join_env.t) (var : variable)
    (ty ty' : flambda_type)
    (relations relations' : Join_env.inverse_relations) :
  state_valid s0 ->
  state_holds s0 ->
  relations_hold s0 relations ->
  exists_in_target s0 var = true ->
  type_in_target kind_in_target_env s0 ty ->
  (∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
     (valuation : env) (v : value),
     Join_env.joined_env s0 index = Some use ->
     typing_env_satisfied_strictly statics valuation use ->
     lookup var (target_valuation s0 index use valuation) = Some v ->
     type_satisfied statics (target_valuation s0 index use valuation) v ty) ->
  (ty' = ty ∧ relations' = relations)
  ∨ Join_env.recover_inverse_relations relations (Name_var var) ty
    = Some (ty', relations') ->
  Join_env.add_equation_in_target_env s0 var ty' = Some s2 ->
  later s0 s2 ∧ state_holds s2 ∧ relations_hold s2 relations'.
Proof.
  intros Hvalid0 Hholds0 Hrelations Hvar Htarget Hupper Hcase Hadd.
  assert (Htags_v :
            ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
              (valuation : env) (v : value),
              Join_env.joined_env s0 index = Some use ->
              tags_at s0 use statics valuation ->
              lookup var (target_valuation s0 index use valuation) = Some v ->
              ∀ (tag : Z) (fields : list value),
                block_of statics v = Some (tag, fields) -> is_tag tag).
  { intros index use statics valuation v Huse Htags Hv tag fields Hblock.
    destruct (target_value_of_simple s0 index use valuation var v Hholds0 Huse
                Hv) as [simple [Hscope Heval]].
    exact (Htags simple v tag fields Hscope Heval Hblock). }
  assert (Hequation : equation_holds s0 var ty').
  { split; [ exact Hvar | ]. split.
    - intros y Hy. destruct Hcase as [[-> _] | Hrecover].
      + left. exact (Htarget y Hy).
      + destruct (recover_free_variables _ _ _ _ _ y Hrecover Hy)
          as [Hin | ->]; left; [ exact (Htarget y Hin) | exact Hvar ].
    - intros index use statics valuation v Huse Hsatisfied Htags Hv.
      pose proof (Hupper _ _ _ _ _ Huse Hsatisfied Hv) as Hty.
      destruct Hcase as [[-> _] | Hrecover]; [ exact Hty | ].
      exact (recover_keeps_meaning _ _ _ _ _ statics _ v Hrecover Hv
               (Htags_v _ _ _ _ _ Huse Htags Hv) Hty). }
  destruct (add_equation_holds s0 s2 var ty' Hvalid0 Hholds0 Hequation Hadd)
    as [Hlater2 Hholds2].
  split; [ exact Hlater2 | ]. split; [ exact Hholds2 | ].
  intros r relation n Hrecorded.
  destruct Hcase as [[_ ->] | Hrecover].
  - exact (later_relations_hold _ _ _ Hlater2 Hrelations r relation n
             Hrecorded).
  - destruct (recovered_relations_hold _ _ _ _ _ r relation n Hrecover
                Hrecorded) as [Hold | [-> [Hr Hnew]]].
    + exact (later_relations_hold _ _ _ Hlater2 Hrelations r relation n Hold).
    + exists var. split; [ reflexivity | ].
      split; [ exact (later_exists _ _ _ Hlater2 Hvar) | ].
      split; [ exact (later_exists _ _ _ Hlater2 (Htarget r Hr)) | ].
      intros index use statics valuation v Huse Hsatisfied Htags Hv.
      rewrite (later_env _ _ _ Hlater2) in Huse.
      rewrite (later_target_valuation s0 s2 index use valuation var Hlater2
                 (or_introl Hvar)) in Hv.
      rewrite (later_target_valuation s0 s2 index use valuation r Hlater2
                 (or_introl (Htarget r Hr))).
      pose proof (later_tags_at _ _ _ _ _ Hlater2 Htags) as Htags0.
      exact (Hnew statics _ v (Htags_v _ _ _ _ _ Huse Htags0 Hv)
               (Hupper _ _ _ _ _ Huse Hsatisfied Hv)).
Qed.

(* The step of a round. *)
Definition round_step
    (acc : option (Join_env.inverse_relations * Join_env.t))
    (binding : variable * Keyed.finite_map Index.t flambda_type)
    : option (Join_env.inverse_relations * Join_env.t) :=
  let* '(relations, env) := acc in
  let var := fst binding in
  let types := snd binding in
  let* '(joined, env) := jt env (Index.bindings types) in
  match joined with
  | Or_unknown.Unknown => Some (relations, env)
  | Or_unknown.Known ty =>
    let* '(ty, relations) :=
      if Join_env.recover_relations_on choices env var types
      then Join_env.recover_inverse_relations relations (Name_var var) ty
      else Some (ty, relations)
    in
    let* env := Join_env.add_equation_in_target_env env var ty in
    Some (relations, env)
  end.

Lemma round_step_none :
  ∀ bindings : list (variable * Keyed.finite_map Index.t flambda_type),
    List.fold_left round_step bindings None = None.
Proof.
  induction bindings as [ | binding bindings IH]; [ reflexivity | exact IH ].
Qed.

Lemma round_fold_holds :
  ∀ (bindings : list (variable * Keyed.finite_map Index.t flambda_type))
    (s s' : Join_env.t) (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       In (var, types) bindings -> types_hold s var types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    relations_hold s relations ->
    List.fold_left round_step bindings (Some (relations, s))
    = Some (relations', s') ->
    later s s' ∧ state_holds s' ∧ relations_hold s' relations'.
Proof.
  induction bindings as [ | [var types] bindings IH];
    intros s s' relations relations' Htypes Htrue Hinvariant Hkinded
      Hrelations Hfold.
  - cbn in Hfold. injection Hfold as <- <-.
    split; [ exact (later_refl s (proj1 Htrue)) | ].
    split; [ exact (proj1 (proj2 Htrue)) | exact Hrelations ].
  - cbn [List.fold_left] in Hfold.
    destruct (round_step (Some (relations, s)) (var, types))
      as [[relations1 s1] | ] eqn:Hstep;
      [ | rewrite round_step_none in Hfold; discriminate ].
    assert (Hfirst :
              later s s1 ∧ state_holds s1 ∧ relations_hold s1 relations1).
    { unfold round_step in Hstep. cbn [fst snd] in Hstep.
      destruct (jt s (Index.bindings types)) as [[joined s0] | ] eqn:Hjoin;
        [ | discriminate ].
      destruct (join_call_holds s s0 var types joined Htrue Hinvariant Hkinded
                  (Htypes var types (or_introl eq_refl)) Hjoin)
        as [Hlater0 [Hholds0 Hknown]].
      assert (Hvar0 : exists_in_target s0 var = true).
      { exact (later_exists _ _ _ Hlater0
                 (proj1 (Htypes var types (or_introl eq_refl)))). }
      pose proof (later_relations_hold _ _ _ Hlater0 Hrelations)
        as Hrelations0.
      destruct joined as [ty | ].
      - destruct (Hknown ty eq_refl) as [Htarget Hupper].
        destruct (Join_env.recover_relations_on choices s0 var types).
        + destruct (Join_env.recover_inverse_relations relations
                      (Name_var var) ty)
            as [[ty' relations0] | ] eqn:Hrecover; [ | discriminate ].
          destruct (Join_env.add_equation_in_target_env s0 var ty')
            as [s2 | ] eqn:Hadd; [ | discriminate ].
          injection Hstep as <- <-.
          destruct (known_equation_holds s0 s2 var ty ty' relations
                      relations0 (proj1 (proj2 Hlater0)) Hholds0 Hrelations0
                      Hvar0 Htarget Hupper (or_intror Hrecover) Hadd)
            as [Hlater2 [Hholds2 Hrelations2]].
          split; [ exact (later_trans _ _ _ Hlater0 Hlater2) | ].
          split; [ exact Hholds2 | exact Hrelations2 ].
        + destruct (Join_env.add_equation_in_target_env s0 var ty)
            as [s2 | ] eqn:Hadd; [ | discriminate ].
          injection Hstep as <- <-.
          destruct (known_equation_holds s0 s2 var ty ty relations relations
                      (proj1 (proj2 Hlater0)) Hholds0 Hrelations0 Hvar0
                      Htarget Hupper (or_introl (conj eq_refl eq_refl)) Hadd)
            as [Hlater2 [Hholds2 Hrelations2]].
          split; [ exact (later_trans _ _ _ Hlater0 Hlater2) | ].
          split; [ exact Hholds2 | exact Hrelations2 ].
      - injection Hstep as <- <-.
        split; [ exact Hlater0 | ].
        split; [ exact Hholds0 | exact Hrelations0 ]. }
    destruct Hfirst as [Hlater1 [Hholds1 Hrelations1]].
    destruct (IH s1 s' relations1 relations'
                (fun v t Hin =>
                   later_types_hold _ _ _ _ Hlater1
                     (Htypes v t (or_intror Hin)))
                (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
                (later_envs_invariant _ _ Hlater1 Hinvariant)
                (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hfold)
      as [Hlater' [Hholds' Hrelations']].
    split; [ exact (later_trans _ _ _ Hlater1 Hlater') | ].
    split; [ exact Hholds' | exact Hrelations' ].
Qed.

Lemma round_holds (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations) :
  (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
     lookup var equations = Some types -> types_hold s var types) ->
  state_true source s ->
  envs_invariant s ->
  envs_kinded s ->
  relations_hold s relations ->
  Join_env.n_way_join_round choices jt s equations relations
  = Some (relations', s') ->
  later s s' ∧ state_holds s' ∧ relations_hold s' relations'.
Proof.
  intros Htypes Htrue Hinvariant Hkinded Hrelations Hround.
  apply (round_fold_holds (compact equations) s s' relations relations');
    [ | exact Htrue | exact Hinvariant | exact Hkinded | exact Hrelations
      | exact Hround ].
  intros var types Hin. apply Htypes. apply in_compact. exact Hin.
Qed.

Lemma delayed_equations_hold :
  ∀ (fuel : nat) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       lookup var equations = Some types -> types_hold s var types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    relations_hold s relations ->
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    later s s' ∧ state_holds s' ∧ relations_hold s' relations'.
Proof.
  induction fuel as [ | fuel IH];
    intros s s' equations relations relations' Htypes Htrue Hinvariant Hkinded
      Hrelations Hloop; [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as <- <-.
    split; [ exact (later_refl s (proj1 Htrue)) | ].
    split; [ exact (proj1 (proj2 Htrue)) | exact Hrelations ].
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    destruct (round_holds s s1 equations relations relations1 Htypes Htrue
                Hinvariant Hkinded Hrelations Hround)
      as [Hlater1 [Hholds1 Hrelations1]].
    assert (Htypes1 :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var (Join_env.new_equations_in_joined_envs s1 s)
                = Some types ->
                types_hold s1 var types).
    { intros var types Hvar.
      unfold Join_env.new_equations_in_joined_envs in Hvar.
      rewrite lookup_filter_bindings in Hvar.
      destruct (lookup var (Join_env.types_in_joined_envs s1))
        as [types1 | ] eqn:Hlookup; [ | discriminate ].
      destruct (negb (is_bound var (Join_env.types_in_joined_envs s)));
        [ | discriminate ].
      injection Hvar as <-.
      exact (proj1 (proj2 Hholds1) var types1 Hlookup). }
    destruct (IH s1 s' _ relations1 relations' Htypes1
                (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
                (later_envs_invariant _ _ Hlater1 Hinvariant)
                (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hloop)
      as [Hlater' [Hholds' Hrelations']].
    split; [ exact (later_trans _ _ _ Hlater1 Hlater') | ].
    split; [ exact Hholds' | exact Hrelations' ].
Qed.

(* The loop over the variables that the join has made. *)
Lemma loop_holds (p : Join_env.parameters) :
  expand_const_sound p ->
  ∀ (depth fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    (∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
       lookup x existentials = Some definition ->
       lookup x (Join_env.definitions_of_existentials s) = Some definition) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    relations_hold s relations ->
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    later s s' ∧ state_holds s' ∧ relations_hold s' relations'.
Proof.
  intros Hconst.
  induction depth as [ | depth IH];
    intros fuel s s' relations relations' existentials Hdefinitions Htrue
      Hinvariant Hkinded Hrelations Hloop;
    cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as <- <-.
    split; [ exact (later_refl s (proj1 Htrue)) | ].
    split; [ exact (proj1 (proj2 Htrue)) | exact Hrelations ].
  - destruct (Keyed.is_empty existentials).
    { injection Hloop as <- <-.
      split; [ exact (later_refl s (proj1 Htrue)) | ].
      split; [ exact (proj1 (proj2 Htrue)) | exact Hrelations ]. }
    destruct (Join_env.equations_of_existentials p s (compact existentials))
      as [equations | ] eqn:Hequations; [ | discriminate ].
    destruct (Join_env.n_way_join_delayed_equations fuel choices jt s
                relations equations)
      as [[relations1 s1] | ] eqn:Hdelayed; [ | discriminate ].
    assert (Htypes :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types -> types_hold s var types).
    { intros var types Hvar.
      destruct (equations_of_existentials_lookup p s _ _ Hequations _ _ Hvar)
        as [simples [kind [Hin Hfind]]].
      apply in_compact in Hin.
      exact (existential_types_hold p s var simples kind types Hconst
               (proj1 Htrue) (proj1 (proj2 Htrue)) Hinvariant Hkinded
               (Hdefinitions _ _ Hin) Hfind). }
    destruct (delayed_equations_hold fuel s s1 equations relations relations1
                Htypes Htrue Hinvariant Hkinded Hrelations Hdelayed)
      as [Hlater1 [Hholds1 Hrelations1]].
    assert (Hdefinitions1 :
              ∀ (x : variable)
                (definition : Simples_in_joined_envs.t * flambda_kind),
                lookup x (Join_env.new_definitions_of_existentials s1 s)
                = Some definition ->
                lookup x (Join_env.definitions_of_existentials s1)
                = Some definition).
    { intros x definition Hx.
      unfold Join_env.new_definitions_of_existentials in Hx.
      rewrite lookup_filter_bindings in Hx.
      destruct (lookup x (Join_env.definitions_of_existentials s1))
        as [definition1 | ]; [ | discriminate ].
      destruct (negb (is_bound x (Join_env.definitions_of_existentials s)));
        [ | discriminate ].
      exact Hx. }
    destruct (IH fuel s1 s' relations1 relations' _ Hdefinitions1
                (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
                (later_envs_invariant _ _ Hlater1 Hinvariant)
                (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hloop)
      as [Hlater' [Hholds' Hrelations']].
    split; [ exact (later_trans _ _ _ Hlater1 Hlater') | ].
    split; [ exact Hholds' | exact Hrelations' ].
Qed.

(* The phases of the join before the variables of the target are
   defined. *)
Lemma first_phases_hold (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  expand_const_sound p ->
  (∀ x : variable, knows source x -> x ∈ avoid) ->
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     knows use x ->
     x ∈ avoid) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     typing_env_invariant use) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     typing_env_kinded use) ->
  levels_found envs ->
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable)
     (kind : flambda_kind) (ty : flambda_type)
     (binding_time_and_mode : Binding_time.With_name_mode.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     simple_of_kind use (Var x) kind ->
     Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types source)
     = Some (ty, binding_time_and_mode) ->
     FlambdaTypeGrammar.kind ty = kind) ->
  Join_env.join_aliases_into_bindings p fuel
    (Join_env.create envs
       (Bindings_in_target_env.from_source_env avoid source))
    envs
  = Some s1 ->
  Join_env.n_way_join_delayed_equations fuel choices jt s1 []
    (Join_env.new_equations_in_joined_envs s1
       (Join_env.create envs
          (Bindings_in_target_env.from_source_env avoid source)))
  = Some (relations2, s2) ->
  Join_env.n_way_join_loop (Join_env.join_depth p) p fuel choices jt s2
    relations2
    (Join_env.new_definitions_of_existentials s2
       (Join_env.create envs
          (Bindings_in_target_env.from_source_env avoid source)))
  = Some (relations3, s3) ->
  state_true source s3
  ∧ relations_hold s3 relations3
  ∧ grows
      (Join_env.create envs
         (Bindings_in_target_env.from_source_env avoid source))
      s3.
Proof.
  intros Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded Hlevels Hkinds
    Haliases Hdelayed Hloop.
  set (s0 :=
         Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid source)) in *.
  pose proof (created_state_true avoid source envs Hsource Huses) as Htrue0.
  fold s0 in Htrue0.
  pose proof (created_state_kinded avoid source envs Hkinds) as Hstate0.
  fold s0 in Hstate0.
  assert (Hknown0 :
            ∀ x : variable, exists_in_target s0 x = true -> knows source x).
  { intros x Hx. rewrite kind_in_target_env_exists in Hx.
    unfold s0, Join_env.create, Bindings_in_target_env.exists_in_target_env,
      Bindings_in_target_env.from_source_env in Hx.
    cbn in Hx. rewrite !orb_false_r in Hx. exact Hx. }
  destruct (join_aliases_true p fuel source s0 s1 Hfresh Heqb Htrue0 Hstate0
              Hinvariant Hkinded Hlevels Hknown0 Haliases)
    as [Htrue1 Hgrows1].
  assert (Hlater1 : later s0 s1).
  { split; [ exact (proj1 Htrue0) | ]. split; [ exact (proj1 Htrue1) | ].
    split; [ exact Hgrows1 | ].
    rewrite (proj2 (proj2 Htrue1)). symmetry. exact (proj2 (proj2 Htrue0)). }
  assert (Hinvariant0 : envs_invariant s0) by exact Hinvariant.
  assert (Hkinded0 : envs_kinded s0) by exact Hkinded.
  assert (Htypes1 :
            ∀ (var : variable)
              (types : Keyed.finite_map Index.t flambda_type),
              lookup var (Join_env.new_equations_in_joined_envs s1 s0)
              = Some types ->
              types_hold s1 var types).
  { intros var types Hvar.
    unfold Join_env.new_equations_in_joined_envs in Hvar.
    rewrite lookup_filter_bindings in Hvar.
    destruct (lookup var (Join_env.types_in_joined_envs s1))
      as [types1 | ] eqn:Hlookup; [ | discriminate ].
    destruct (negb (is_bound var (Join_env.types_in_joined_envs s0)));
      [ | discriminate ].
    injection Hvar as <-.
    exact (proj1 (proj2 (proj1 (proj2 Htrue1))) var types1 Hlookup). }
  assert (Hrelations1 : relations_hold s1 []).
  { intros r relation n [recorded [Hlookup _]]. discriminate. }
  destruct (delayed_equations_hold fuel s1 s2 _ [] relations2 Htypes1 Htrue1
              (later_envs_invariant _ _ Hlater1 Hinvariant0)
              (later_envs_kinded _ _ Hlater1 Hkinded0) Hrelations1 Hdelayed)
    as [Hlater2 [Hholds2 Hrelations2]].
  pose proof (later_trans _ _ _ Hlater1 Hlater2) as Hlater02.
  assert (Hdefinitions2 :
            ∀ (x : variable)
              (definition : Simples_in_joined_envs.t * flambda_kind),
              lookup x (Join_env.new_definitions_of_existentials s2 s0)
              = Some definition ->
              lookup x (Join_env.definitions_of_existentials s2)
              = Some definition).
  { intros x definition Hx.
    unfold Join_env.new_definitions_of_existentials in Hx.
    rewrite lookup_filter_bindings in Hx.
    destruct (lookup x (Join_env.definitions_of_existentials s2))
      as [definition2 | ]; [ | discriminate ].
    destruct (negb (is_bound x (Join_env.definitions_of_existentials s0)));
      [ | discriminate ].
    exact Hx. }
  destruct (loop_holds p Hconst (Join_env.join_depth p) fuel s2 s3 relations2
              relations3 _ Hdefinitions2
              (later_true source s1 s2 Hlater2 Hholds2
                 (proj2 (proj2 Htrue1)))
              (later_envs_invariant _ _ Hlater02 Hinvariant0)
              (later_envs_kinded _ _ Hlater02 Hkinded0) Hrelations2 Hloop)
    as [Hlater3 [Hholds3 Hrelations3]].
  pose proof (later_trans _ _ _ Hlater02 Hlater3) as Hlater03.
  split; [ exact (later_true source s0 s3 Hlater03 Hholds3
                    (proj2 (proj2 Htrue0))) | ].
  split; [ exact Hrelations3 | exact (proj1 (proj2 (proj2 Hlater03))) ].
Qed.

End Rounds.


Theorem delayed_equations_true : delayed_equations_true_statement.
Proof.
  intros source jt choices fuel s s' equations relations relations' Hjt Htypes
    Htrue Hinvariant Hkinded Hrelations Hloop.
  destruct (delayed_equations_hold source jt choices Hjt fuel s s' equations
              relations relations' Htypes Htrue Hinvariant Hkinded Hrelations
              Hloop) as [Hlater [Hholds Hrelations']].
  split; [ exact (later_true source s s' Hlater Hholds (proj2 (proj2 Htrue)))
         | ].
  split; [ exact Hrelations' | exact (proj1 (proj2 (proj2 Hlater))) ].
Qed.

Theorem loop_true : loop_true_statement.
Proof.
  intros source jt choices p depth fuel s s' relations relations' existentials
    Hjt Hconst Hdefinitions Htrue Hinvariant Hkinded Hrelations Hloop.
  destruct (loop_holds source jt choices Hjt p Hconst depth fuel s s'
              relations relations' existentials Hdefinitions Htrue Hinvariant
              Hkinded Hrelations Hloop) as [Hlater [Hholds Hrelations']].
  split; [ exact (later_true source s s' Hlater Hholds (proj2 (proj2 Htrue)))
         | ].
  split; [ exact Hrelations' | exact (proj1 (proj2 (proj2 Hlater))) ].
Qed.

Theorem first_phases_true : first_phases_true_statement.
Proof.
  intros source jt choices p fuel avoid envs s1 s2 s3 relations2 relations3
    Hjt Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded Hlevels Hkinds
    Haliases Hdelayed Hloop.
  exact (first_phases_hold source jt choices Hjt p fuel avoid envs s1 s2 s3
           relations2 relations3 Hfresh Heqb Hconst Hsource Huses Hinvariant
           Hkinded Hlevels Hkinds Haliases Hdelayed Hloop).
Qed.
