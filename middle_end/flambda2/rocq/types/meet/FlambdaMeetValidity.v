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


(* The contracts of the meet, with what it is for a static environment
   to be valid as a parameter.

   Nothing here is a statement, and nothing here is to be reviewed
   for what it says of the compiler: the four definitions below are
   made from four definitions of FlambdaMeetSpec.v and
   FlambdaMeetStepsSpec.v by a tactic that puts a parameter where
   they have [statics_valid] (FlambdaClosuresSpec.v), and at
   [statics_valid] each is the definition it was made from, by
   computation ([..._at_standard] below).

   What it is for. Every clause of those contracts that speaks of
   values supposes a valid static environment. The proofs of the
   functions of the meet never look into that supposition: they hand
   it on, and to the facts about closures that need it. So they hold
   as they are for any predicate on static environments that implies
   validity ([validity]). At [statics_valid] they are the signed
   statements. At the predicate that is false of every static
   environment ([no_validity]) the clauses about values say nothing,
   and what is left is the half that speaks of environments and
   types only: what a meet keeps. That is how
   FlambdaMeetKeepsSpec.v gets, for a join that has only the half of
   its contract that is not about values, that the meet keeps what
   it keeps. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypingEnv FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv
  FlambdaMeet FlambdaMeetSpec FlambdaMeetStepsSpec FlambdaMeetScopesSpec
  FlambdaMeetHeadsSpec FlambdaMeetRowLikeSpec FlambdaMeetFuelSpec.
Import ListNotations.

(* A predicate on static environments that implies validity. *)
Class validity : Type := {
  valid : static_env -> Prop;
  valid_statics : ∀ statics : static_env, valid statics -> statics_valid statics
}.

(* Validity itself. The proofs that are made for any [validity] are
   read at this one for the signed statements. *)
#[global] Instance standard_validity : validity | 100 :=
  {| valid := statics_valid; valid_statics := fun _ H => H |}.

(* No static environment. *)
Definition no_validity : validity :=
  {| valid := fun _ => False;
     valid_statics := fun _ (H : False) => match H with end |}.

(* [meet_expanded_head_sound], [join_scoped_envs_sound]
   (FlambdaMeetSpec.v), [meet_call_ok] and [part_meet_ok]
   (FlambdaMeetStepsSpec.v), with [valid] where they have
   [statics_valid]. *)
Definition meet_expanded_head_sound_for {V : validity}
    : meet_expanded_head -> Prop :=
  ltac:(let t :=
          eval cbv delta [meet_expanded_head_sound] in meet_expanded_head_sound
        in
        let f := eval pattern statics_valid in t in
        lazymatch f with
        | ?F _ => exact (F valid)
        end).

Definition join_scoped_envs_sound_for {V : validity}
    : join_scoped_envs -> Prop :=
  ltac:(let t :=
          eval cbv delta [join_scoped_envs_sound] in join_scoped_envs_sound
        in
        let f := eval pattern statics_valid, meet_expanded_head_sound in t in
        lazymatch f with
        | ?F _ _ => exact (F valid meet_expanded_head_sound_for)
        end).

Definition meet_call_ok_for {V : validity} :=
  ltac:(let t := eval cbv delta [meet_call_ok] in (@meet_call_ok) in
        let f := eval pattern statics_valid in t in
        lazymatch f with
        | ?F _ => exact (F valid)
        end).

Definition part_meet_ok_for {V : validity} :=
  ltac:(let t := eval cbv delta [part_meet_ok] in (@part_meet_ok) in
        let f := eval pattern (@meet_call_ok) in t in
        lazymatch f with
        | ?F _ => exact (F (@meet_call_ok_for V))
        end).

Arguments meet_call_ok_for {V A}%_type_scope fits satisfied menv a1 a2 result.
Arguments part_meet_ok_for {V A}%_type_scope fits satisfied meet.

(* Four definitions of the other files of statements are made of those:
   [meet_disjunction_right] (FlambdaMeetScopesSpec.v),
   [type_part_meet_ok] (FlambdaMeetHeadsSpec.v),
   [meet_row_like_for_closures_right] (FlambdaMeetRowLikeSpec.v) and
   [value_heads_right] (FlambdaMeetFuelSpec.v). The same is done to
   them. *)
Definition meet_disjunction_right_for {V : validity} : Prop :=
  ltac:(let t :=
          eval cbv delta [meet_disjunction_right] in meet_disjunction_right
        in
        let f :=
          eval pattern statics_valid, (@part_meet_ok), (@meet_call_ok),
            meet_expanded_head_sound, join_scoped_envs_sound
          in t
        in
        lazymatch f with
        | ?F _ _ _ _ _ =>
          exact
            (F valid (@part_meet_ok_for V) (@meet_call_ok_for V)
               meet_expanded_head_sound_for join_scoped_envs_sound_for)
        end).

Definition type_part_meet_ok_for {V : validity} :=
  ltac:(let t := eval cbv delta [type_part_meet_ok] in (@type_part_meet_ok) in
        let f := eval pattern (@part_meet_ok) in t in
        lazymatch f with
        | ?F _ => exact (F (@part_meet_ok_for V))
        end).

Arguments type_part_meet_ok_for {V X}%_type_scope to_type v meet.

Definition meet_row_like_for_closures_right_for {V : validity} : Prop :=
  ltac:(let t :=
          eval cbv delta [meet_row_like_for_closures_right] in
            meet_row_like_for_closures_right
        in
        let f :=
          eval pattern (@type_part_meet_ok), meet_expanded_head_sound,
            join_scoped_envs_sound
          in t
        in
        lazymatch f with
        | ?F _ _ _ =>
          exact
            (F (@type_part_meet_ok_for V) meet_expanded_head_sound_for
               join_scoped_envs_sound_for)
        end).

Definition value_heads_right_for {V : validity} :=
  ltac:(let t := eval cbv delta [value_heads_right] in value_heads_right in
        let f := eval pattern (@part_meet_ok) in t in
        lazymatch f with
        | ?F _ => exact (F (@part_meet_ok_for V))
        end).

(* A statement of the files of the meet, with the eight definitions
   above where it has those of the signed files, and [valid] where it
   has [statics_valid]. *)
Ltac generalise_validity X :=
  let t := eval cbv delta [X] in X in
  let f :=
    eval pattern statics_valid, (@part_meet_ok), (@meet_call_ok),
      meet_expanded_head_sound, join_scoped_envs_sound,
      meet_disjunction_right, (@type_part_meet_ok),
      meet_row_like_for_closures_right, value_heads_right
    in t
  in
  lazymatch f with
  | ?F _ _ _ _ _ _ _ _ _ =>
    lazymatch goal with
    | V : validity |- _ =>
      let g :=
        eval cbv beta in
          (F (@valid V) (@part_meet_ok_for V) (@meet_call_ok_for V)
             (@meet_expanded_head_sound_for V) (@join_scoped_envs_sound_for V)
             (@meet_disjunction_right_for V) (@type_part_meet_ok_for V)
             (@meet_row_like_for_closures_right_for V)
             (@value_heads_right_for V))
      in
      exact g
    end
  end.

Notation "'for_validity' X" := (ltac:(generalise_validity X))
  (at level 10, X at level 9, only parsing).

(* At validity itself, the four are the signed definitions. *)
Lemma meet_expanded_head_sound_at_standard (meet_head : meet_expanded_head) :
  meet_expanded_head_sound_for (V := standard_validity) meet_head
  = meet_expanded_head_sound meet_head.
Proof. reflexivity. Qed.

Lemma join_scoped_envs_sound_at_standard (join : join_scoped_envs) :
  join_scoped_envs_sound_for (V := standard_validity) join
  = join_scoped_envs_sound join.
Proof. reflexivity. Qed.

Lemma meet_call_ok_at_standard {A : Type} (fits : Typing_env.t -> A -> Prop)
    (satisfied : static_env -> env -> A -> Prop) (menv : Meet_env.t)
    (a1 a2 : A) (result : Meet_result.t A) :
  meet_call_ok_for (V := standard_validity) fits satisfied menv a1 a2 result
  = meet_call_ok fits satisfied menv a1 a2 result.
Proof. reflexivity. Qed.

Lemma part_meet_ok_at_standard {A : Type} (fits : Typing_env.t -> A -> Prop)
    (satisfied : static_env -> env -> A -> Prop)
    (meet : Meet_env.t -> A -> A -> option (Meet_result.t A)) :
  part_meet_ok_for (V := standard_validity) fits satisfied meet
  = part_meet_ok fits satisfied meet.
Proof. reflexivity. Qed.

(* With no valid static environment, every meet of expanded types has
   the contract that speaks of values. *)
Lemma meet_expanded_head_sound_at_none (meet_head : meet_expanded_head) :
  meet_expanded_head_sound_for (V := no_validity) meet_head.
Proof.
  unfold meet_expanded_head_sound_for.
  intros statics valuation menv expanded1 expanded2 v result. cbv zeta.
  intros Hwf Hvalid. destruct Hvalid.
Qed.
