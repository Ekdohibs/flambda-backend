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


(* For the proofs of the statements of FlambdaMeetRenameSpec.v: the
   variables of a level and the meet of two alternatives under a
   renaming. Nothing here needs review.

   The laws that need the meet of less fuel or the join are for two
   meets and two joins that one renaming relates
   ([meet_expanded_head_related], [join_related_at]): the primed one
   is given the renamed arguments. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeRenameSpec FlambdaTypeRenameProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaTypingEnv
  FlambdaEnvRenameSpec FlambdaEnvRenameProofs FlambdaMeetGrammar
  FlambdaExpandHead FlambdaProversRenameSpec FlambdaProversRenameProofs
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetRenameSpec
  FlambdaEnvRenameMeetProofs.
Import ListNotations.

(* ======================================================================== *)
(* The environment of the meet                                              *)
(* ======================================================================== *)

Lemma names_to_avoid_renamed (r : renaming) (env : Meet_env.t) :
  Meet_env.names_to_avoid (rename_meet_env r env)
  = map (rename_variable r) (Meet_env.names_to_avoid env).
Proof. reflexivity. Qed.

Lemma typing_env_renamed (r : renaming) (env : Meet_env.t) :
  Meet_env.typing_env (rename_meet_env r env)
  = rename_typing_env r (Meet_env.typing_env env).
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* The variables of a level                                                 *)
(* ======================================================================== *)

Lemma extension_of_level_renamed (r : renaming)
    (level : Typing_env_level.t) :
  extension_of_level (rename_level r level)
  = rename_env_extension r (extension_of_level level).
Proof. exact (level_as_extension_equivariant r level). Qed.

Lemma define_in_types_renamed (choices : meet_choices) (r : renaming)
    (env : Meet_env.t) (var : variable) (kind : flambda_kind) :
  define_in_types choices (rename_meet_env r env) (rename_variable r var)
    kind
  = option_map (rename_meet_env r) (define_in_types choices env var kind).
Proof.
  unfold define_in_types. rewrite typing_env_renamed.
  change (Name_var (rename_variable r var)) with (rename_name r (Name_var var)).
  rewrite (mem_equivariant r None (Meet_env.typing_env env) (Name_var var)).
  destruct (define_again choices
            || negb
                 (Typing_env.mem None (Meet_env.typing_env env)
                    (Name_var var)));
    [ | reflexivity ].
  exact (add_definition_renamed r env (Name_var var) Name_mode.in_types kind).
Qed.

Lemma add_defined_vars_renamed (choices : meet_choices) (r : renaming)
    (env : Meet_env.t) (level : Typing_env_level.t) :
  add_defined_vars choices (rename_meet_env r env) (rename_level r level)
  = option_map (rename_meet_env r) (add_defined_vars choices env level).
Proof.
  unfold add_defined_vars.
  pose proof
    (level_fold_on_defined_vars_equivariant (option Meet_env.t) r
       (option_map (rename_meet_env r))
       (fun var kind env =>
          let* env := env in
          define_in_types choices env var kind)
       (fun var kind env =>
          let* env := env in
          define_in_types choices env var kind)
       level (Some env))
    as Hfold.
  cbn [option_map] in Hfold. rewrite Hfold.
  - destruct (Typing_env_level.fold_on_defined_vars _ level (Some env))
      as [[env' | ] | ]; reflexivity.
  - intros var kind [acc | ]; [ | reflexivity ].
    exact (define_in_types_renamed choices r acc var kind).
Qed.

(* ======================================================================== *)
(* The meet of two alternatives                                             *)
(* ======================================================================== *)

(* The call that has been rewritten to its renamed answer is taken
   apart: an answer that may be none, a result of a meet, an
   environment or bottom. *)
Ltac rn_split :=
  match goal with
  | |- context [option_map _ ?x] =>
    lazymatch x with
    | match _ with _ => _ end => fail
    | Some _ => fail
    | None => fail
    | _ => destruct x eqn:?
    end
  | |- context [rename_meet_result _ _ ?x] => is_var x; destruct x
  | |- context [or_bottom_map _ ?x] => is_var x; destruct x
  end.

#[local] Arguments extension_of_level : simpl never.

(* What is left when the calls have been taken apart: the two sides
   differ by where the renaming is, in what the meets returned. *)
Ltac rn_leaf :=
  repeat match goal with
         | x : meet_return_value _ |- _ => destruct x
         end;
  simpl; try reflexivity.

Definition rename_disjunction {A B : Type} (ra : renaming -> A -> A)
    (rb : renaming -> B -> B) (r : renaming)
    (x : A * B * variant_extensions) : A * B * variant_extensions :=
  (ra r (fst (fst x)), rb r (snd (fst x)),
   rename_variant_extensions r (snd x)).

Section Disjunction.

Variable r : renaming.
Variables join join' : join_scoped_envs.
Variable choices : meet_choices.
Variables m m' : meet_expanded_head.

Hypothesis Hjoin : join_related_at r join join' m m'.
Hypothesis Hm : meet_expanded_head_related r m m'.

Lemma add_extension_of_level_renamed (env : Meet_env.t)
    (level : Typing_env_level.t) :
  Meet_env.add_env_extension_strict (rename_meet_env r env)
    (extension_of_level (rename_level r level)) m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_env_extension_strict env (extension_of_level level) m).
Proof.
  rewrite extension_of_level_renamed.
  apply (add_env_extension_strict_renamed r m m' Hm).
Qed.

Lemma join_two_renamed (cut_after : Scope.t) (env : Meet_env.t)
    (tenv tenv_a tenv_b : Typing_env.t) :
  join' m' cut_after (rename_meet_env r env) (rename_typing_env r tenv)
    [rename_typing_env r tenv_a; rename_typing_env r tenv_b]
  = option_map (rename_meet_env r)
      (join m cut_after env tenv [tenv_a; tenv_b]).
Proof. exact (Hjoin cut_after env tenv [tenv_a; tenv_b]). Qed.

Lemma meet_disjunction_renamed {A B : Type} (ra : renaming -> A -> A)
    (rb : renaming -> B -> B)
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) :
  meet_of_parts_related r (ra r) (ra r) meet_a meet_a' ->
  meet_of_parts_related r (rb r) (rb r) meet_b meet_b' ->
  ra r (bottom_a tt) = bottom_a tt ->
  rb r (bottom_b tt) = bottom_b tt ->
  ∀ (env : Meet_env.t) (a1 : A) (b1 : B) (ext1 : variant_extensions)
    (a2 : A) (b2 : B) (ext2 : variant_extensions),
    meet_disjunction join' choices m' meet_a' meet_b' bottom_a bottom_b
      (rename_meet_env r env) (ra r a1) (rb r b1)
      (rename_variant_extensions r ext1) (ra r a2) (rb r b2)
      (rename_variant_extensions r ext2)
    = option_map (rename_meet_result r (rename_disjunction ra rb r))
        (meet_disjunction join choices m meet_a meet_b bottom_a bottom_b env
           a1 b1 ext1 a2 b2 ext2).
Proof.
  intros Ha Hb Hbottom_a Hbottom_b env a1 b1 ext1 a2 b2 ext2.
  unfold meet_disjunction. rewrite enter_scope_renamed.
  destruct (Meet_env.enter_scope env) as [[join_scope initial_tenv] env']
    eqn:Henter.
  cbn [fst snd]. unfold Meet_env.final_typing_env.
  destruct ext1 as [ | when_a1 when_b1]; destruct ext2 as [ | when_a2 when_b2];
    cbn [rename_variant_extensions].
  all: repeat
         (simpl;
          repeat first
            [ rewrite (add_env_extension_strict_renamed r m m' Hm)
            | rewrite Ha | rewrite Hb
            | rewrite cut_equivariant
            | rewrite with_names_to_avoid_renamed
            | rewrite add_defined_vars_renamed
            | rewrite add_extension_of_level_renamed
            | rewrite join_two_renamed
            | rewrite level_is_empty_equivariant ];
          try rn_split).
  all: rewrite ?extension_of_level_renamed.
  all: repeat match goal with
              | |- context [if ?c then _ else _] =>
                match type of c with
                | bool => destruct c
                end
              end.
  all: rn_leaf.
  all: unfold rename_disjunction; simpl; rewrite ?Hbottom_a, ?Hbottom_b;
    try reflexivity.
Qed.

End Disjunction.
