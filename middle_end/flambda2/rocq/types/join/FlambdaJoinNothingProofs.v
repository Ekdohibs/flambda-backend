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


(* The proofs of the statements of FlambdaJoinNothingSpec.v. This file
   needs no review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaJoinEnv FlambdaJoinEnvSpec FlambdaJoinEnvProofs FlambdaMeetEnv
  FlambdaMeetSpec FlambdaJoinMeet FlambdaJoinNothingSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* The level cut from an environment one scope further, after the
   scope that it had, is empty. *)
Lemma cut_incremented (tenv : Typing_env.t) :
  Typing_env.cut (Typing_env.increment_scope tenv)
    (Typing_env.current_scope tenv)
  = Some Typing_env_level.empty.
Proof.
  unfold Typing_env.cut, Typing_env.increment_scope, Typing_env.current_scope.
  cbn [Typing_env.current_level Typing_env.prev_levels
         Typing_env.One_level.scope Typing_env.One_level.level
         Typing_env.cut_loop].
  unfold Scope.next.
  destruct (Z.leb_spec
              (Typing_env.One_level.scope (Typing_env.current_level tenv) + 1)
              (Typing_env.One_level.scope (Typing_env.current_level tenv)))
    as [Hle | _]; [ lia | ].
  rewrite Z.ltb_irrefl. reflexivity.
Qed.

(* The environments joined: each with no equation, or none if the
   environment of the meet is bottom. *)
Lemma cut_for_joins_of_nothing (tenv : Typing_env.t) :
  ∀ envs : list (Index.t * Typing_env.t),
    (∀ binding : Index.t * Typing_env.t,
       binding ∈ envs -> snd binding = Typing_env.increment_scope tenv) ->
    Join_env.cut_for_joins (Typing_env.current_scope tenv) envs
    = Some
        (if Typing_env.is_bottom tenv
         then []
         else
           map
             (fun binding : Index.t * Typing_env.t =>
                (fst binding, (snd binding, [])))
             envs).
Proof.
  induction envs as [ | [index use] envs IH]; intros Hall.
  - destruct (Typing_env.is_bottom tenv); reflexivity.
  - cbn [Join_env.cut_for_joins].
    rewrite (IH (fun binding Hin => Hall binding (or_intror Hin))).
    pose proof (Hall (index, use) (or_introl eq_refl)) as Huse.
    cbn [snd] in Huse. subst use.
    change (Typing_env.is_bottom (Typing_env.increment_scope tenv))
      with (Typing_env.is_bottom tenv).
    destruct (Typing_env.is_bottom tenv); [ reflexivity | ].
    rewrite cut_incremented. reflexivity.
Qed.

(* No name has an equation in every level, where no level has one. *)
Lemma no_names_with_equations (envs : Joined_envs.t) :
  (∀ (index : Index.t) (use : Typing_env.t)
     (equations : Keyed.finite_map name flambda_type),
     (index, (use, equations)) ∈ envs -> equations = []) ->
  Join_env.names_with_equations_everywhere envs = [].
Proof.
  intros Hall. unfold Join_env.names_with_equations_everywhere.
  destruct (Keyed.compact Index.equal envs)
    as [ | [index [use equations]] others] eqn:Hcompact; [ reflexivity | ].
  assert (Hin : (index, (use, equations)) ∈ Keyed.compact Index.equal envs).
  { rewrite Hcompact. left. reflexivity. }
  apply (keyed_in_compact Index.equal index_equal_decides) in Hin.
  apply (keyed_lookup_in Index.equal index_equal_decides) in Hin.
  rewrite (Hall index use equations Hin). reflexivity.
Qed.

(* A typing environment that is bottom is the one made bottom. *)
Lemma make_bottom_of_bottom (tenv : Typing_env.t) :
  Typing_env.is_bottom tenv = true -> Typing_env.make_bottom tenv = tenv.
Proof.
  destruct tenv. cbn [Typing_env.is_bottom]. intros ->. reflexivity.
Qed.

(* The uses, numbered: one at least, each the environment one scope
   further. *)
Lemma numbered_all (tenv : Typing_env.t) (scoped_envs : list Typing_env.t) :
  (∀ scoped : Typing_env.t,
     scoped ∈ scoped_envs -> scoped = Typing_env.increment_scope tenv) ->
  ∀ binding : Index.t * Typing_env.t,
    binding ∈ Index.number scoped_envs ->
    snd binding = Typing_env.increment_scope tenv.
Proof.
  intros Hall [index use] Hin. unfold Index.number in Hin.
  apply in_combine_r in Hin. exact (Hall use Hin).
Qed.

Theorem model_join_scoped_envs_of_nothing
  : model_join_scoped_envs_of_nothing_statement.
Proof.
  intros join_choices join_depth fuel env scoped_envs Hfuel Hnonempty Hall.
  set (tenv := Meet_env.typing_env env) in *.
  unfold model_join_scoped_envs, Join_env.cut_and_n_way_join.
  rewrite (cut_for_joins_of_nothing tenv _ (numbered_all tenv _ Hall)).
  destruct (Typing_env.is_bottom tenv) eqn:Hbottom.
  - (* Bottom: no environment is joined. *)
    exists (Meet_env.make_bottom env). split.
    + unfold Meet_env.make_bottom, Meet_env.map_typing_env.
      cbn [Meet_env.typing_env]. exact (make_bottom_of_bottom tenv Hbottom).
    + intros mh. reflexivity.
  - (* One environment at least, each with no equation. *)
    destruct scoped_envs as [ | scoped scoped_envs];
      [ exfalso; exact (Hnonempty eq_refl) | ].
    set (avoid :=
           Join_env.names_to_avoid meet_env_functions env
           ++ Join_env.variables_of_env tenv
           ++ flat_map Join_env.variables_of_env (scoped :: scoped_envs)).
    exists (Meet_env.with_names_to_avoid env avoid).
    split; [ reflexivity | ]. intros mh.
    set (envs :=
           map
             (fun binding : Index.t * Typing_env.t =>
                (fst binding, (snd binding, [])))
             (Index.number (scoped :: scoped_envs))).
    assert (Hnames : Join_env.names_with_equations_everywhere envs = []).
    { apply no_names_with_equations. intros index use equations Hin.
      unfold envs in Hin. apply in_map_iff in Hin.
      destruct Hin as [binding [Heq _]]. injection Heq as _ _ <-.
      reflexivity. }
    cbv zeta. unfold Join_env.cut_and_n_way_join0. fold avoid envs.
    change (Keyed.is_empty envs) with false. cbv iota.
    unfold Join_env.join_aliases_into_bindings. rewrite Hnames.
    cbn [List.fold_left].
    destruct (fuel_of_env fuel) as [ | fuel_of_env];
      [ exfalso; exact (Hfuel eq_refl) | ].
    destruct join_depth; reflexivity.
Qed.

Theorem model_join_scoped_envs_no_fuel
  : model_join_scoped_envs_no_fuel_statement.
Proof.
  intros join_choices join_depth fuel mh env scoped_envs Hfuel Hbottom
    Hnonempty Hall.
  set (tenv := Meet_env.typing_env env) in *.
  unfold model_join_scoped_envs, Join_env.cut_and_n_way_join.
  rewrite (cut_for_joins_of_nothing tenv _ (numbered_all tenv _ Hall)).
  rewrite Hbottom.
  destruct scoped_envs as [ | scoped scoped_envs];
    [ exfalso; exact (Hnonempty eq_refl) | ].
  set (envs :=
         map
           (fun binding : Index.t * Typing_env.t =>
              (fst binding, (snd binding, [])))
           (Index.number (scoped :: scoped_envs))).
  assert (Hnames : Join_env.names_with_equations_everywhere envs = []).
  { apply no_names_with_equations. intros index use equations Hin.
    unfold envs in Hin. apply in_map_iff in Hin.
    destruct Hin as [binding [Heq _]]. injection Heq as _ _ <-.
    reflexivity. }
  cbv zeta. unfold Join_env.cut_and_n_way_join0. fold envs.
  change (Keyed.is_empty envs) with false. cbv iota.
  unfold Join_env.join_aliases_into_bindings. rewrite Hnames.
  cbn [List.fold_left]. rewrite Hfuel. reflexivity.
Qed.
